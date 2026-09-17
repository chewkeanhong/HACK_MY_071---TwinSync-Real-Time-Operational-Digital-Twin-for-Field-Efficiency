"""Recorded simulation states, and the guard that stops a stale one reaching the stage.

The guided demo's Prev/Next buttons restore these, so two properties matter. The big
immutable objects must be *shared* rather than copied, or sixteen beats would cost
hundreds of megabytes and restore into a different world than the server is running. And
a recording that no longer matches the code must be refused: restoring one would not
crash, it would quietly put a different scenario on screen mid-presentation.

A stand-in stands for the Simulation here. `checkpoints` only reaches for `world`,
`coverage`, `intelligence.risk` and `detectors`, and building a real one costs seconds of
world loading for no extra coverage of this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from twinsync import checkpoints


class Heavy:
    """Stands in for the world, the coverage engine, a booster, an ONNX session."""

    def __init__(self, name: str):
        self.name = name


@dataclass
class FakeDetector:
    model: object | None = None


@dataclass
class FakeIntelligence:
    risk: object = None


@dataclass
class FakeSim:
    world: object
    coverage: object
    intelligence: FakeIntelligence
    detectors: dict = field(default_factory=dict)
    t: float = 0.0
    payload: list = field(default_factory=list)


def make_sim(shared: dict) -> FakeSim:
    return FakeSim(
        world=shared["world"],
        coverage=shared["coverage"],
        intelligence=FakeIntelligence(risk=shared["risk"]),
        detectors={"KL-01": FakeDetector(model=shared["model"]),
                   "KL-02": FakeDetector(model=shared["model"])},
        t=1200.0,
        payload=[{"incident": "INC-004", "subscribers": 9026}],
    )


@pytest.fixture
def shared():
    return {"world": Heavy("world"), "coverage": Heavy("coverage"),
            "risk": Heavy("risk"), "model": Heavy("onnx")}


# -- sharing the immutables ---------------------------------------------


def test_restored_state_reuses_the_live_world(tmp_path, shared):
    """The restored simulation must point at the server's objects, not copies of them."""
    checkpoints.save(make_sim(shared), tmp_path / "beat-00.pkl")
    template = make_sim(shared)

    restored = checkpoints.load(tmp_path / "beat-00.pkl", template)

    assert restored.world is shared["world"]
    assert restored.coverage is shared["coverage"]
    assert restored.intelligence.risk is shared["risk"]
    assert restored.detectors["KL-01"].model is shared["model"]


def test_mutable_state_is_restored_not_shared(tmp_path, shared):
    """Everything else is a genuine copy, or two jumps would corrupt each other."""
    checkpoints.save(make_sim(shared), tmp_path / "beat-00.pkl")
    template = make_sim(shared)

    restored = checkpoints.load(tmp_path / "beat-00.pkl", template)
    restored.payload[0]["subscribers"] = 1

    assert restored.t == 1200.0
    assert template.payload[0]["subscribers"] == 9026


def test_the_shared_objects_are_not_written_to_disk(tmp_path, shared):
    """Sixteen beats x the whole world is the mistake this avoids."""
    checkpoints.save(make_sim(shared), tmp_path / "beat-00.pkl")
    blob = (tmp_path / "beat-00.pkl").read_bytes()

    # The tags travel; the objects behind them do not.
    assert b"coverage" in blob
    assert b"Heavy" not in blob


# -- refusing a stale recording -----------------------------------------


def write_recording(tmp_path, beats, fingerprint="deadbeef"):
    directory = tmp_path / checkpoints.CHECKPOINT_DIR_NAME
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(len(beats)):
        (directory / f"beat-{index:02d}.pkl").write_bytes(b"x")
    (directory / checkpoints.MANIFEST_NAME).write_text(
        json.dumps({"fingerprint": fingerprint, "beats": beats}), encoding="utf-8")
    return directory


@pytest.fixture
def scenario_dir(tmp_path):
    (tmp_path / "scenario.json").write_text('{"seed": 42}', encoding="utf-8")
    return tmp_path


def test_a_matching_recording_is_usable(scenario_dir):
    beats = [{"t_s": 0}, {"t_s": 45}]
    write_recording(scenario_dir, beats,
                    checkpoints.fingerprint(scenario_dir, beats))

    manifest = checkpoints.load_manifest(scenario_dir, beats)

    assert manifest.usable
    assert manifest.time_for(1) == 45


def test_missing_recording_is_reported_not_raised(scenario_dir):
    manifest = checkpoints.load_manifest(scenario_dir, [{"t_s": 0}])

    assert not manifest.usable
    assert "bake_checkpoints" in manifest.reason


def test_a_changed_scenario_invalidates_the_recording(scenario_dir):
    beats = [{"t_s": 0}, {"t_s": 45}]
    write_recording(scenario_dir, beats,
                    checkpoints.fingerprint(scenario_dir, beats))
    (scenario_dir / "scenario.json").write_text('{"seed": 7}', encoding="utf-8")

    manifest = checkpoints.load_manifest(scenario_dir, beats)

    assert not manifest.usable
    assert "stale" in manifest.reason


def test_retimed_beats_invalidate_the_recording(scenario_dir):
    beats = [{"t_s": 0}, {"t_s": 45}]
    write_recording(scenario_dir, beats,
                    checkpoints.fingerprint(scenario_dir, beats))

    manifest = checkpoints.load_manifest(scenario_dir, [{"t_s": 0}, {"t_s": 90}])

    assert not manifest.usable


def test_rewording_a_caption_does_not_force_a_rebake(scenario_dir):
    """A ten-minute re-bake for a typo fix would just train people to skip it."""
    before = checkpoints.fingerprint(scenario_dir, [{"t_s": 0, "title": "A city"}])
    after = checkpoints.fingerprint(scenario_dir, [{"t_s": 0, "title": "A city, rewritten"}])

    assert before == after


def test_an_added_beat_invalidates_the_recording(scenario_dir):
    beats = [{"t_s": 0}, {"t_s": 45}]
    write_recording(scenario_dir, beats,
                    checkpoints.fingerprint(scenario_dir, beats))

    manifest = checkpoints.load_manifest(scenario_dir, beats + [{"t_s": 90}])

    assert not manifest.usable


def test_a_deleted_file_invalidates_the_recording(scenario_dir):
    beats = [{"t_s": 0}, {"t_s": 45}]
    directory = write_recording(scenario_dir, beats,
                                checkpoints.fingerprint(scenario_dir, beats))
    (directory / "beat-01.pkl").unlink()

    manifest = checkpoints.load_manifest(scenario_dir, beats)

    assert not manifest.usable
    assert "missing" in manifest.reason
