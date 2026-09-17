"""The annualised saving, and the demo track that narrates it.

Two things a judge reads straight off the screen, neither of which the rest of the suite
touched:

* the headline RM figure, which is a *projection* -- measured per-incident savings times
  two stated assumptions. The dashboard re-projects it from the committed A/B result
  rather than simulating a second arm, so the two paths have to agree exactly or the
  slide and the screen quote different numbers.
* `data/demo.json`, which drives the guided demo. A typo there is invisible until the
  pitch is running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twinsync.metrics import (
    DEFAULT_INCIDENT_RATE,
    DEFAULT_SITES,
    Comparison,
    per_incident_from_results,
    project_annual,
)

from .test_metrics import build

DATA = Path(__file__).resolve().parents[1] / "data"


# -- the projection ------------------------------------------------------


def comparison() -> Comparison:
    """A baseline arm that costs more than the TwinSync arm on every axis."""
    baseline = build("baseline", truck_rolls=8, travel_distance_m=20_000.0,
                     subscriber_minutes_lost=2000.0)
    twinsync = build("twinsync", truck_rolls=4, travel_distance_m=10_000.0,
                     subscriber_minutes_lost=1000.0)
    return Comparison(baseline=baseline, twinsync=twinsync)


def test_projection_scales_linearly_with_fleet_size():
    unit = comparison().per_incident()
    small = project_annual(unit, sites=1000, incidents_per_site_per_year=4.0)
    large = project_annual(unit, sites=4000, incidents_per_site_per_year=4.0)
    assert large["cost_saved_myr"] == pytest.approx(4 * small["cost_saved_myr"], rel=1e-6)
    assert large["truck_rolls_avoided"] == pytest.approx(
        4 * small["truck_rolls_avoided"], rel=1e-6)


def test_projection_scales_linearly_with_fault_rate():
    unit = comparison().per_incident()
    slow = project_annual(unit, sites=2000, incidents_per_site_per_year=2.0)
    fast = project_annual(unit, sites=2000, incidents_per_site_per_year=6.0)
    assert fast["km_saved"] == pytest.approx(3 * slow["km_saved"], rel=1e-6)


def test_projection_echoes_its_own_assumptions():
    """The multipliers are arguments, not findings, so they ride along with the answer.

    Quoting the product without them is the dishonest version of this number.
    """
    projected = project_annual(comparison().per_incident(), sites=3210,
                               incidents_per_site_per_year=2.5)
    assert projected["assumed_sites"] == 3210
    assert projected["assumed_incidents_per_site_per_year"] == 2.5
    assert projected["implied_incidents_per_year"] == round(3210 * 2.5)


def test_annualised_delegates_to_the_shared_projection():
    """RunComparison.annualised and the standalone projection must not drift apart."""
    compare = comparison()
    assert compare.annualised(sites=750, incidents_per_site_per_year=3.0) == \
        project_annual(compare.per_incident(), sites=750,
                       incidents_per_site_per_year=3.0)


def test_per_incident_survives_a_round_trip_through_results_json():
    """The dashboard reads a serialised run; it must recover the same unit savings.

    This is the join that matters: `/api/metrics` re-projects from `results.json`
    instead of running a second dispatch arm, so if this drifts the dashboard quietly
    disagrees with the results table.
    """
    compare = comparison()
    recovered = per_incident_from_results(compare.as_dict())
    for key, value in compare.per_incident().items():
        assert recovered[key] == pytest.approx(value, rel=1e-9)


def test_recovery_falls_back_when_the_exact_block_is_missing():
    """Older results.json files predate the exact block and must still re-project.

    The fallback divides rounded totals, so it is only good to the two decimals
    `as_dict` writes -- close enough for ringgit, visibly off for kilograms of CO2.
    That looseness is the reason the exact block exists, and is asserted here rather
    than discovered later.
    """
    compare = comparison()
    legacy = {k: v for k, v in compare.as_dict().items() if k != "per_incident"}
    recovered = per_incident_from_results(legacy)

    assert recovered["cost_saved_myr"] == pytest.approx(
        compare.per_incident()["cost_saved_myr"], rel=1e-9)
    assert recovered["co2_saved_kg"] == pytest.approx(
        compare.per_incident()["co2_saved_kg"], rel=1e-3)


# -- the committed artifacts --------------------------------------------


def test_committed_results_reproject_to_their_own_headline():
    """Re-projecting data/results.json must land back on the number it already carries.

    The README and the KPI tile quote this figure from two different code paths.
    """
    results = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    stored = results["annualised"]
    reprojected = project_annual(
        per_incident_from_results(results),
        sites=stored["assumed_sites"],
        incidents_per_site_per_year=stored["assumed_incidents_per_site_per_year"])

    assert reprojected["cost_saved_myr"] == stored["cost_saved_myr"]
    assert reprojected["truck_rolls_avoided"] == stored["truck_rolls_avoided"]


def test_projection_defaults_are_the_ones_the_docs_quote():
    assert (DEFAULT_SITES, DEFAULT_INCIDENT_RATE) == (2000, 4.0)


# -- the guided demo track ----------------------------------------------


def demo_track() -> dict:
    return json.loads((DATA / "demo.json").read_text(encoding="utf-8"))


def test_demo_beats_are_ordered_and_complete():
    beats = demo_track()["beats"]
    assert beats, "no beats"
    for beat in beats:
        assert beat["title"].strip()
        assert beat["body"].strip()
        assert isinstance(beat["t_s"], (int, float))
    times = [b["t_s"] for b in beats]
    assert times == sorted(times), "beats must be in chronological order"


def test_demo_beats_use_real_view_modes():
    """A typo here silently leaves the pane on whatever it was showing before."""
    for beat in demo_track()["beats"]:
        if "view" in beat:
            assert beat["view"] in {"2d", "3d", "split"}, beat["view"]


def test_demo_beats_only_focus_towers_that_exist():
    towers = json.loads((DATA / "towers.geojson").read_text(encoding="utf-8"))
    known = {f["properties"]["id"] for f in towers["features"]}
    for beat in demo_track()["beats"]:
        if beat.get("focus"):
            assert beat["focus"] in known, f"{beat['focus']} is not a site in this AOI"


def test_demo_beats_land_after_the_events_they_narrate():
    """Each act's beat must fire *after* the fault it describes, not before it.

    Authoring a beat a few seconds early is easy and produces the worst possible demo
    failure: the caption announces something the screen has not done yet.
    """
    scenario = json.loads((DATA / "scenario.json").read_text(encoding="utf-8"))
    beats = demo_track()["beats"]

    for fault in scenario["faults"]:
        narrating = [b for b in beats if b.get("focus") == fault["tower"]]
        if not narrating:
            continue
        assert min(b["t_s"] for b in narrating) >= fault["start_s"], (
            f"a beat narrates {fault['tower']} before it fails")


def test_demo_track_fits_inside_the_scenario():
    scenario = json.loads((DATA / "scenario.json").read_text(encoding="utf-8"))
    last = max(b["t_s"] for b in demo_track()["beats"])
    assert last < scenario["duration_s"], "the track outlives the scenario it narrates"
