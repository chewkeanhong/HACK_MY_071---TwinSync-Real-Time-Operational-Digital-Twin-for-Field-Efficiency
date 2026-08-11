from __future__ import annotations

from edge.intelligence import IntelligenceLayerSimulator

from .conftest import make_world


def _world_with_towers():
    return make_world(
        specs=[{"id": "b1", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 30.0}],
        towers=[
            {"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T2", "x": 120.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T3", "x": 900.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
        ],
    )


def test_st_dbscan_simulator_is_deterministic_per_members_and_time_bucket():
    sim = IntelligenceLayerSimulator(_world_with_towers())
    a = sim.simulate_st_dbscan({"T2", "T1"}, now_s=179.9)
    b = sim.simulate_st_dbscan({"T1", "T2"}, now_s=150.0)
    c = sim.simulate_st_dbscan({"T1", "T2"}, now_s=260.0)

    assert a.members == ["T1", "T2"]
    assert a.cluster_id == b.cluster_id
    assert a.cluster_id != c.cluster_id
    assert a.model == "st-dbscan-simulated"


def test_lightgbm_simulator_scores_severity_and_impact_higher():
    sim = IntelligenceLayerSimulator(_world_with_towers())
    mild = sim.simulate_lightgbm_risk(
        severity="degraded",
        subscribers=300,
        critical_count=0,
        minutes_open=2.0,
        sla_minutes=60.0,
    )
    severe = sim.simulate_lightgbm_risk(
        severity="down",
        subscribers=9000,
        critical_count=3,
        minutes_open=55.0,
        sla_minutes=60.0,
    )

    assert 0.0 <= mild.score <= 100.0
    assert 0.0 <= severe.score <= 100.0
    assert severe.score > mild.score
    assert severe.band in {"medium", "high"}
    assert severe.model == "lightgbm-simulated"
