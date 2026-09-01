"""The intelligence adapter: localisation, risk, and the degraded path.

The clustering algorithm itself is covered in test_stdbscan.py. What matters here is
that the adapter wires it up correctly, and that a missing model artifact produces an
honest degraded mode rather than a crash or a silent lie about which model ran.
"""

from __future__ import annotations

from edge.intelligence import IntelligenceLayer
from twinsync.risk import HEURISTIC_MODEL, AssetProfile, RiskScorer, band_for

from .conftest import make_world


def _world_with_towers():
    return make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 30.0}],
        towers=[
            {"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T2", "x": 120.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T3", "x": 9000.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
        ],
    )


def test_adapter_groups_a_real_cascade_and_isolates_an_unrelated_fault():
    layer = IntelligenceLayer(_world_with_towers(), models_dir="does-not-exist")

    first = layer.localise("T1", "amplifier_degradation", 100.0)
    assert first.is_noise is True

    second = layer.localise("T2", "amplifier_degradation", 160.0)
    assert second.is_noise is False
    assert second.members == ["T1", "T2"]

    # 9 km away: same fault kind, same minute, still not the same incident.
    far = layer.localise("T3", "amplifier_degradation", 200.0)
    assert far.is_noise is True
    assert far.members == ["T3"]


def test_release_drops_a_resolved_alarm():
    layer = IntelligenceLayer(_world_with_towers(), models_dir="does-not-exist")
    layer.localise("T1", "amplifier_degradation", 100.0)
    layer.release("T1")
    assert layer.localise("T2", "amplifier_degradation", 160.0).is_noise is True


def test_missing_artifact_degrades_honestly():
    """No booster on disk must mean a working demo that says it is running degraded."""
    layer = IntelligenceLayer(_world_with_towers(), models_dir="does-not-exist")
    assert layer.degraded is True
    assert HEURISTIC_MODEL in layer.model_source
    assert "st-dbscan-v1" in layer.model_source

    result = layer.score_risk("T1", severity="down", subscribers=9000,
                              critical_count=3, minutes_open=55.0, sla_minutes=60.0)
    assert result.model == HEURISTIC_MODEL
    assert 0.0 <= result.score <= 100.0


def test_heuristic_still_ranks_severity_and_impact():
    scorer = RiskScorer()          # no booster
    world = _world_with_towers()
    tower = world.tower("T1")

    mild = scorer.score(tower, severity="degraded", subscribers=300, critical_count=0,
                        minutes_open=2.0, sla_minutes=60.0)
    severe = scorer.score(tower, severity="down", subscribers=9000, critical_count=3,
                          minutes_open=55.0, sla_minutes=60.0)

    assert severe.score > mild.score
    assert severe.band in {"medium", "high"}


def test_band_thresholds():
    assert band_for(90.0) == "high"
    assert band_for(60.0) == "medium"
    assert band_for(10.0) == "low"


def test_asset_profile_is_stable_across_processes():
    """Derived from a digest, not hash() -- so it survives a restart. See telemetry."""
    a = AssetProfile.for_tower("KL-07")
    b = AssetProfile.for_tower("KL-07")
    assert a == b
    assert AssetProfile.for_tower("KL-08") != a
    assert 1.0 <= a.asset_age_years <= 15.0
    assert 0.35 <= a.load_factor <= 0.95
