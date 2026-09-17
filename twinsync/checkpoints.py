"""Saved simulation states, so the guided demo can jump to a beat.

The scenario is stateful: crews are part-way through routes, detectors carry a learned
baseline, batteries are draining, roads are under water. There is no formula for "the
world at t=1200 s" -- the only way to get there is to run the simulation to it, and that
costs about 80 ms per 0.2 s step, so replaying twenty minutes of scenario takes about ten
minutes of wall clock. Far too slow to do behind a button on stage.

So the run is recorded once, ahead of time, and each beat's state is written to disk.
Jumping is then a read rather than a replay: about a second, and identical to having let
the demo run there, because the simulation is deterministic given its seed.

**What is not written.** The world, the coverage engine, the risk booster and the ONNX
anomaly model are large, immutable and already loaded by whoever is restoring. Writing
them into all sixteen files would cost hundreds of megabytes to say the same thing
sixteen times, so they are swapped for a tag on the way out and swapped back for the
live objects on the way in -- see :class:`_Sharing`.

**Staleness is the real hazard.** A checkpoint is a pickle of live objects, so it is only
valid for the code and scenario that produced it. Restoring a stale one would not crash;
it would quietly put a *different* scenario on screen mid-presentation. Every checkpoint
therefore carries a fingerprint of the scenario, the beat times and the simulation
sources, and :func:`load_manifest` refuses anything that does not match the running code.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_DIR_NAME = "checkpoints"
MANIFEST_NAME = "manifest.json"

# Sources whose behaviour decides what the simulation looks like at a given instant. A
# change to any of them invalidates every recorded state.
FINGERPRINT_SOURCES = (
    "twinsync/sim.py",
    "twinsync/dispatch.py",
    "twinsync/routing.py",
    "twinsync/weather.py",
    "twinsync/world.py",
    "twinsync/coverage.py",
    "twinsync/priority.py",
    "twinsync/risk.py",
    "twinsync/stdbscan.py",
    "twinsync/checkpoints.py",
    "edge/detector.py",
    "edge/telemetry.py",
    "edge/intelligence.py",
)


def fingerprint(data_dir: Path, beats: list[dict], *, root: Path | None = None) -> str:
    """Identify the scenario *and* the code that produced a set of checkpoints."""
    root = root or Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    digest.update((data_dir / "scenario.json").read_bytes())
    # Only the beat times matter: rewording a caption must not force a ten-minute re-bake.
    digest.update(json.dumps([b["t_s"] for b in beats], sort_keys=True).encode())
    for name in FINGERPRINT_SOURCES:
        path = root / name
        digest.update(path.read_bytes() if path.exists() else b"")
    return digest.hexdigest()[:16]


class _Sharing:
    """Swaps the big immutable objects for tags, and back again.

    Pickle calls ``persistent_id`` for every object it is about to write; returning a
    tag makes it write the tag instead of walking into the object. ``persistent_load``
    does the reverse on the way in, handing back whatever the live process already holds.
    """

    def __init__(self, sim):
        self.objects = {
            "world": sim.world,
            "coverage": sim.coverage,
            "risk": sim.intelligence.risk,
        }
        for detector in sim.detectors.values():
            if detector.model is not None:
                self.objects["anomaly-model"] = detector.model
                break
        self.tags = {id(obj): tag for tag, obj in self.objects.items()}


class _Pickler(pickle.Pickler):
    def __init__(self, file, sharing: _Sharing):
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self._sharing = sharing

    def persistent_id(self, obj):
        return self._sharing.tags.get(id(obj))


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, sharing: _Sharing):
        super().__init__(file)
        self._sharing = sharing

    def persistent_load(self, tag):
        try:
            return self._sharing.objects[tag]
        except KeyError:  # pragma: no cover - a corrupt or foreign checkpoint
            raise pickle.UnpicklingError(f"checkpoint refers to unknown object {tag!r}")


def save(sim, path: Path) -> int:
    """Write one simulation state. Returns the size on disk, in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        _Pickler(handle, _Sharing(sim)).dump(sim)
    return path.stat().st_size


def load(path: Path, template) -> object:
    """Read one simulation state back, reattaching `template`'s shared objects.

    `template` is any live Simulation -- the server's current one will do. It is only
    read from, never advanced, and the object returned is independent of it.
    """
    with path.open("rb") as handle:
        return _Unpickler(handle, _Sharing(template)).load()


@dataclass(frozen=True)
class Manifest:
    """What was recorded, and whether it still describes the running code."""

    directory: Path
    beats: list[dict]
    stale: bool
    reason: str = ""

    @property
    def usable(self) -> bool:
        return not self.stale and bool(self.beats)

    def path_for(self, index: int) -> Path:
        return self.directory / f"beat-{index:02d}.pkl"

    def time_for(self, index: int) -> float:
        return float(self.beats[index]["t_s"])


def load_manifest(data_dir: Path, beats: list[dict]) -> Manifest:
    """Find the recorded states, and check they match the running code.

    Never raises: a missing or stale recording disables jumping and says why, which is a
    better stage failure than a confident jump into a scenario that no longer exists.
    """
    directory = data_dir / CHECKPOINT_DIR_NAME
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        return Manifest(directory, [], True,
                        "no checkpoints recorded -- run scripts/bake_checkpoints.py")
    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return Manifest(directory, [], True, f"unreadable checkpoint manifest: {exc}")

    if saved.get("fingerprint") != fingerprint(data_dir, beats):
        return Manifest(directory, [], True,
                        "checkpoints are stale (the scenario or the simulation changed) "
                        "-- re-run scripts/bake_checkpoints.py")
    recorded = saved.get("beats", [])
    if len(recorded) != len(beats):
        return Manifest(directory, [], True,
                        f"checkpoints cover {len(recorded)} beats but the track has "
                        f"{len(beats)} -- re-run scripts/bake_checkpoints.py")
    missing = [i for i in range(len(recorded))
               if not (directory / f"beat-{i:02d}.pkl").exists()]
    if missing:
        return Manifest(directory, [], True,
                        f"checkpoint files missing for beats {missing} -- re-run "
                        "scripts/bake_checkpoints.py")
    return Manifest(directory, recorded, False)


def write_manifest(directory: Path, data_dir: Path, beats: list[dict],
                   recorded: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MANIFEST_NAME).write_text(
        json.dumps({"fingerprint": fingerprint(data_dir, beats),
                    "beats": recorded}, indent=2),
        encoding="utf-8")
