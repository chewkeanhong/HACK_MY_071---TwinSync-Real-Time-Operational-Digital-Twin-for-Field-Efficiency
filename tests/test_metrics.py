"""The KPIs that go on the results slide.

metrics.py had no tests at all, which is uncomfortable for the module that produces
every number in the pitch. These check the derived quantities against hand-computed
fixtures so an arithmetic slip cannot quietly change a headline figure.
"""

from __future__ import annotations

import pytest

from twinsync.metrics import (
    CO2_KG_PER_L,
    EVENT_BYTES,
    FUEL_L_PER_KM,
    RAW_SAMPLE_BYTES,
    TRUCK_ROLL_COST_MYR,
    Comparison,
    RunMetrics,
)


def build(label: str, **overrides) -> RunMetrics:
    spec = dict(
        incidents_raised=4, incidents_resolved=4,
        resolution_minutes=[10.0, 20.0, 30.0, 40.0],
        detection_minutes=[1.0, 1.0, 1.0, 1.0],
        truck_rolls=4, subscriber_minutes_lost=1000.0,
        travel_distance_m=10_000.0, driving_seconds=600.0, on_site_seconds=1800.0,
        crew_count=2, elapsed_seconds=3600.0, total_subscribers=1000,
    )
    spec.update(overrides)
    return RunMetrics(label=label, **spec)


# -- derived quantities -------------------------------------------------


def test_mttr_is_the_mean_of_resolution_times():
    assert build("x").mttr_minutes == pytest.approx(25.0)


def test_mttr_percentiles():
    run = build("x")
    assert run.mttr_percentile(50) == pytest.approx(25.0)
    assert run.mttr_percentile(90) == pytest.approx(37.0)


def test_percentiles_of_an_empty_run_are_zero_not_an_error():
    assert build("x", resolution_minutes=[]).mttr_percentile(90) == 0.0
    assert build("x", resolution_minutes=[]).mttr_minutes == 0.0


def test_mttl_is_separate_from_mttd():
    run = build("x", localisation_minutes=[2.0, 4.0])
    assert run.mean_detection_minutes == pytest.approx(1.0)
    assert run.mean_localisation_minutes == pytest.approx(3.0)


def test_mttl_is_zero_when_nothing_was_localised():
    """The baseline arm has no localiser, and must not report a fabricated MTTL."""
    assert build("x").mean_localisation_minutes == 0.0


def test_travel_fuel_and_co2_chain():
    """10 km -> litres -> kg, each step against the documented constant."""
    run = build("x", travel_distance_m=10_000.0)
    assert run.travel_km == pytest.approx(10.0)
    assert run.fuel_litres == pytest.approx(10.0 * FUEL_L_PER_KM)
    assert run.co2_kg == pytest.approx(10.0 * FUEL_L_PER_KM * CO2_KG_PER_L)


def test_crew_utilisation_uses_elapsed_time_not_a_shift():
    """2 crews x 3600 s available; 600 driving + 1800 on site = 2400 busy."""
    run = build("x")
    assert run.crew_utilisation_pct == pytest.approx(100.0 * 2400.0 / 7200.0)


def test_crew_utilisation_of_an_idle_fleet_is_zero():
    run = build("x", driving_seconds=0.0, on_site_seconds=0.0)
    assert run.crew_utilisation_pct == 0.0


def test_utilisation_without_crews_does_not_divide_by_zero():
    assert build("x", crew_count=0).crew_utilisation_pct == 0.0


def test_sla_uptime():
    """1000 subscribers x 60 min = 60000 subscriber-minutes; 1000 lost."""
    run = build("x")
    assert run.sla_uptime_pct == pytest.approx(100.0 * (1.0 - 1000.0 / 60_000.0))


def test_sla_uptime_is_100_when_nothing_was_lost():
    assert build("x", subscriber_minutes_lost=0.0).sla_uptime_pct == pytest.approx(100.0)


def test_cost_tracks_truck_rolls():
    assert build("x", truck_rolls=3).cost_myr == pytest.approx(3 * TRUCK_ROLL_COST_MYR)


def test_uplink_reduction():
    run = build("x", samples_generated=1000, events_uplinked=10)
    assert run.raw_bytes == 1000 * RAW_SAMPLE_BYTES
    assert run.uplink_bytes == 10 * EVENT_BYTES
    assert run.uplink_reduction == pytest.approx(
        1.0 - (10 * EVENT_BYTES) / (1000 * RAW_SAMPLE_BYTES))


def test_uplink_reduction_with_no_samples_is_zero_not_nan():
    assert build("x", samples_generated=0).uplink_reduction == 0.0


# -- the comparison -----------------------------------------------------


def comparison() -> Comparison:
    baseline = build("today", truck_rolls=4, travel_distance_m=10_000.0,
                     resolution_minutes=[40.0] * 4, subscriber_minutes_lost=1000.0)
    twinsync = build("TwinSync", truck_rolls=3, travel_distance_m=6_000.0,
                     resolution_minutes=[20.0] * 4, subscriber_minutes_lost=400.0)
    return Comparison(baseline=baseline, twinsync=twinsync)


def test_improvements_are_percent_reductions():
    assert comparison().mttr_improvement_pct == pytest.approx(50.0)


def test_savings_are_baseline_minus_twinsync():
    c = comparison()
    assert c.truck_rolls_avoided == 1
    assert c.km_saved == pytest.approx(4.0)
    assert c.subscriber_minutes_saved == pytest.approx(600.0)
    assert c.cost_saved_myr == pytest.approx(TRUCK_ROLL_COST_MYR)


def test_improvement_against_a_zero_baseline_is_zero_not_infinite():
    zero = build("today", resolution_minutes=[])
    assert Comparison(zero, build("TwinSync")).mttr_improvement_pct == 0.0


def test_per_incident_divides_by_resolved_count():
    c = comparison()
    unit = c.per_incident()
    assert unit["truck_rolls_avoided"] == pytest.approx(1 / 4)
    assert unit["km_saved"] == pytest.approx(1.0)


def test_annualisation_multiplies_the_stated_chain():
    """2000 sites x 4 faults = 8000 incidents, times the per-incident saving."""
    c = comparison()
    annual = c.annualised(sites=2000, incidents_per_site_per_year=4.0)
    assert annual["implied_incidents_per_year"] == 8000
    assert annual["truck_rolls_avoided"] == pytest.approx(8000 * 0.25)
    assert annual["km_saved"] == pytest.approx(8000 * 1.0)


def test_annualisation_exposes_its_assumptions():
    """The multipliers must travel with the number, or it is a marketing claim."""
    annual = comparison().annualised()
    assert "assumed_sites" in annual
    assert "assumed_incidents_per_site_per_year" in annual


def test_as_dict_keeps_the_legacy_detection_key():
    """results.json is read by the dashboard; renaming a key silently breaks it."""
    payload = build("x").as_dict()
    assert payload["mean_detection_minutes"] == payload["mttd_minutes"]


def test_render_produces_a_table():
    text = comparison().render()
    for expected in ("MTTD", "MTTL", "MTTR", "CO2", "SLA uptime", "per incident"):
        assert expected in text


# -- the A/B's control group must stay uncontaminated -------------------


def test_baseline_incidents_carry_no_model_tag():
    """The control arm runs no models, and must not claim to.

    `ai_model_source` defaulting to a model name would credit the baseline with the
    thing under test -- a subtle way to make an A/B meaningless.
    """
    from twinsync.dispatch import Incident
    from twinsync.priority import Impact

    import numpy as np
    incident = Incident(id="INC-000", tower_id="T1", severity="down",
                        detected_at=0.0, impact=Impact(), xy=np.zeros(2))
    assert incident.ai_model_source == "none"
    assert incident.ai_cluster_id is None
    assert incident.ai_localised_at is None
    assert incident.ai_risk_factors == []
