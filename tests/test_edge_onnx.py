"""The edge detector and its exported model.

`edge/detector.py` is the most algorithmically substantial file in the project and had
no tests at all. These cover both the model artifact and the detector that consumes it,
including the fallback path -- which is worth more than it looks, because the fallback
only ever runs on a machine where something has already gone wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edge.detector import METRICS, AnomalyModel, EdgeDetector
from edge.telemetry import FAULT_PROFILES, Fault, TowerTelemetry

MODELS = Path(__file__).resolve().parents[1] / "models"
SAMPLE_HZ = 10.0


@pytest.fixture(scope="module")
def model():
    loaded = AnomalyModel.shared(MODELS)
    if loaded is None:
        pytest.skip("no exported model -- run scripts/train_edge_model.py")
    return loaded


def run_tower(tower_id: str, seconds: float, *, fault: Fault | None = None,
              use_model=None) -> tuple[EdgeDetector, list]:
    telemetry = TowerTelemetry(tower_id, seed=42)
    detector = EdgeDetector(tower_id, seed=42, model=use_model)
    verdicts = []
    for step in range(int(seconds * SAMPLE_HZ)):
        t = step / SAMPLE_HZ
        active = fault if (fault and t >= fault.start_s) else None
        verdicts.append(detector.observe(telemetry.sample(t, active)))
    return detector, verdicts


# -- the artifact ------------------------------------------------------


def test_artifact_loads_and_declares_its_contract(model):
    assert model.threshold > 0.0
    assert model.input_name == "z_scores"
    assert model.meta["input_features"] == list(METRICS)
    assert model.meta["shipped_artifact"].endswith(".onnx")


def test_healthy_z_scores_score_below_threshold(model):
    """Small z-scores are ordinary, and must not trip the confirmation stage."""
    rng = np.random.default_rng(0)
    healthy = np.abs(rng.normal(0.0, 0.8, size=(200, len(METRICS))))
    scores = [model.score(row) for row in healthy]
    assert np.mean(np.array(scores) > model.threshold) < 0.10


def test_anomalous_z_scores_score_above_threshold(model):
    """A metric twenty standard deviations out is not something the model has seen."""
    spike = np.zeros(len(METRICS))
    spike[2] = 20.0
    assert model.score(spike) > model.threshold


def test_int8_was_rejected_for_a_recorded_reason(model):
    """Guards the decision, so a future retrain cannot silently ship a broken artifact.

    INT8 quantisation of a ~700-parameter network is both larger than FP32 and noisy
    enough to swamp the decision threshold. If a future run finds otherwise the meta
    will say so and this test will need updating deliberately.
    """
    assert model.meta["shipped_artifact"] == "edge_anomaly_fp32.onnx"
    assert model.meta["int8_rejected_reason"]
    assert model.meta["fp32"]["max_drift"] < model.meta["int8"]["max_drift"]


# -- the detector ------------------------------------------------------


def test_detector_uses_the_model_when_available(model):
    detector = EdgeDetector("T", model=model)
    assert detector.engine == "onnx"


def test_detector_falls_back_when_the_model_is_absent():
    """A missing artifact must degrade, not crash. This is the demo-day safety net."""
    detector, verdicts = run_tower("FB", 30.0, use_model=False)
    assert detector.engine == "forest"
    assert detector._forest is not None, "fallback forest should be fitted after warmup"
    assert all(v.state == "healthy" for v in verdicts)


def test_shared_returns_none_for_a_missing_directory():
    assert AnomalyModel.shared(Path("no-such-models-dir")) is None


def test_healthy_tower_raises_no_alarm(model):
    """Five minutes of ordinary operation, including the diurnal swing. No pages."""
    _, verdicts = run_tower("HEALTHY", 300.0, use_model=model)
    assert not any(v.changed for v in verdicts)
    assert all(v.state == "healthy" for v in verdicts)


@pytest.mark.parametrize("profile", sorted(FAULT_PROFILES))
def test_every_fault_profile_is_detected(profile, model):
    """All four failure modes, each caught within a few seconds of onset."""
    onset = 60.0
    fault = Fault(tower_id="F", profile=profile, start_s=onset)
    _, verdicts = run_tower("F", 180.0, fault=fault, use_model=model)

    changes = [(i / SAMPLE_HZ, v) for i, v in enumerate(verdicts) if v.changed]
    assert changes, f"{profile} was never detected"

    detected_at, verdict = changes[0]
    latency = detected_at - onset
    assert 0.0 < latency < 15.0, f"{profile} took {latency:.1f}s to detect"
    assert verdict.state in {"degraded", "down"}
    assert verdict.reasons, "a state change must explain itself"


def test_power_failure_is_reported_as_down(model):
    """Hard limits exist so a dead site is not merely 'degraded'."""
    fault = Fault(tower_id="P", profile="power_failure", start_s=60.0)
    _, verdicts = run_tower("P", 180.0, fault=fault, use_model=model)
    assert any(v.state == "down" for v in verdicts)


def test_confirmation_stage_is_duty_cycled(model):
    """The expensive path must not run on every sample -- that is the whole design."""
    detector, _ = run_tower("DUTY", 120.0, use_model=model)
    assert detector.confirm_evaluations > 0
    assert detector.confirm_evaluations < detector.samples_seen / 2


def test_verdict_exposes_the_deprecated_alias(model):
    _, verdicts = run_tower("ALIAS", 20.0, use_model=model)
    assert verdicts[-1].forest_score == verdicts[-1].confirm_score
    assert verdicts[-1].confirm_engine in {"onnx", "forest"}


def test_baseline_does_not_learn_the_fault(model):
    """The classic adaptive-threshold failure: absorbing the fault as the new normal."""
    fault = Fault(tower_id="L", profile="amplifier_degradation", start_s=60.0)
    detector, verdicts = run_tower("L", 400.0, fault=fault, use_model=model)
    # Long after onset the site must still be reported unwell.
    assert verdicts[-1].state in {"degraded", "down"}
    assert detector.state in {"degraded", "down"}
