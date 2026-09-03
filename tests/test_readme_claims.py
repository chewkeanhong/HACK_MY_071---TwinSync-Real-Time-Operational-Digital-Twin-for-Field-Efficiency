"""Every headline number in the README must still match the artifact it came from.

The README is the judge-facing surface, and its numbers are quoted from four generated
files: `data/results.json`, `data/bench_edge.json`, `data/bench_twin.json`,
`data/ndvi.json` and the two model metadata files. Nothing stops those regenerating with
different values and leaving the prose behind -- and a stale number in the README is
worse than no number, because it reads as a fabrication rather than a mistake.

So the expected strings are *built from the artifacts* and asserted to appear in the
document. That catches drift in both directions: rerunning a benchmark without updating
the text, and editing the text away from what was measured.

Deliberately not exhaustive. These are the figures a judge actually probes.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def artifact(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def normalise(text: str) -> str:
    """Collapse the things that are typography rather than fact.

    Prose wraps, so "80 KB\nper tower" and "80 KB per tower" are the same claim. The
    README also uses a real minus sign and the odd non-breaking space.
    """
    return re.sub(r"\s+", " ", text.replace("\u2212", "-").replace("\u00a0", " "))


def assert_in_readme(fragment: str, why: str) -> None:
    assert normalise(fragment) in normalise(README), \
        f"README no longer says {fragment!r} -- {why}"


# -- the A/B results table ----------------------------------------------


def test_results_table_matches_the_measured_run():
    r = artifact("data/results.json")
    b, t = r["baseline"], r["twinsync"]

    for value, why in [
        (f"{b['mttd_minutes']} min", "baseline MTTD"),
        (f"**{t['mttd_minutes']} min**", "TwinSync MTTD"),
        (f"{b['mttr_minutes']} min", "baseline MTTR"),
        (f"**{t['mttr_minutes']} min**", "TwinSync MTTR"),
        (f"{b['mttr_p90_minutes']} min", "baseline p90"),
        (f"{t['mttr_p90_minutes']} min", "TwinSync p90 -- the deliberately worse one"),
        (f"{b['subscriber_minutes_lost']:,}", "baseline subscriber-minutes"),
        (f"**{t['subscriber_minutes_lost']:,}**", "TwinSync subscriber-minutes"),
    ]:
        assert_in_readme(value, why)


def test_truck_roll_counts_and_cost_match():
    r = artifact("data/results.json")
    b, t = r["baseline"], r["twinsync"]
    assert_in_readme(f"| truck rolls | {b['truck_rolls']} | **{t['truck_rolls']}** |",
                     "the truck-roll row")
    assert_in_readme(f"RM {b['cost_myr']:,.0f}", "baseline cost of truck rolls")
    assert_in_readme(f"**RM {t['cost_myr']:,.0f}**", "TwinSync cost of truck rolls")


def test_uplink_reduction_matches():
    """Stated in binary units, matching what the dashboard's own formatter shows."""
    t = artifact("data/results.json")["twinsync"]
    assert_in_readme(f"{t['raw_bytes'] / 1048576:.1f} MiB raw", "raw telemetry volume")
    assert_in_readme(f"{t['uplink_bytes'] / 1024:.0f} KiB", "uplinked volume")
    assert_in_readme(f"{100 * t['uplink_reduction']:.1f} %", "the reduction percentage")


# -- the ROI projection --------------------------------------------------


def test_annualised_projection_matches():
    a = artifact("data/results.json")["annualised"]
    assert_in_readme(f"**{a['truck_rolls_avoided']:,}**", "truck rolls avoided per year")
    assert_in_readme(f"**RM {a['cost_saved_myr']:,}**", "the headline ringgit figure")
    assert_in_readme(f"{a['km_saved']:,} km", "distance not driven")
    assert_in_readme(f"{a['co2_saved_tonnes']} t", "CO2 avoided")
    assert_in_readme(f"{a['subscriber_hours_saved']:,}", "subscriber-hours restored")


def test_the_projections_two_assumptions_are_stated():
    """Quoting the product without the multipliers is the dishonest version."""
    a = artifact("data/results.json")["annualised"]
    assert_in_readme(f"{a['assumed_sites']:,} sites", "the assumed network size")
    assert_in_readme(f"{a['assumed_incidents_per_site_per_year']:.0f} faults/site/year",
                     "the assumed fault rate")
    assert_in_readme(f"{a['implied_incidents_per_year']:,} incidents/year",
                     "the implied incident count")


def test_per_incident_savings_match():
    u = artifact("data/results.json")["per_incident"]
    assert_in_readme(f"{u['truck_rolls_avoided']:.2f} truck rolls", "per-incident rolls")
    assert_in_readme(f"RM {u['cost_saved_myr']:.0f}", "per-incident cost")
    assert_in_readme(f"{u['km_saved']:.2f} km", "per-incident distance")
    assert_in_readme(f"{u['subscriber_minutes_saved']:,.0f} subscriber-minutes",
                     "per-incident subscriber-minutes")


# -- the edge benchmark --------------------------------------------------


# The IsolationForest row is quoted to whole microseconds. That is an editorial choice
# rather than drift: a tenth of a microsecond on a 1.9 ms number is noise, while the
# other stages are three orders of magnitude smaller where the decimal carries meaning.
LATENCY_PRECISION = {"isolation_forest": 0}


@pytest.mark.parametrize("stage", ["ewma_z", "mahalanobis", "onnx_fp32",
                                   "isolation_forest", "full_observe"])
def test_edge_latency_table_matches(stage):
    s = artifact("data/bench_edge.json")["stages"][stage]
    dp = LATENCY_PRECISION.get(stage, 1)
    for pct in ("p50", "p95", "p99"):
        assert_in_readme(f"{s[pct + '_us']:.{dp}f} µs", f"{stage} {pct}")


def test_edge_memory_claims_match():
    m = artifact("data/bench_edge.json")["memory"]
    assert_in_readme(f"**{m['model_bytes'] / 1024:.1f} KB**", "model size on disk")
    assert_in_readme(f"**{m['fleet_rss_bytes'] / 1048576:.1f} MB**", "fleet RSS")
    assert_in_readme(f"{m['rss_per_tower_bytes'] / 1024:.0f} KB per tower",
                     "per-tower RSS")


def test_the_126x_speedup_is_arithmetic_on_the_benchmark():
    """The claim that replacing IsolationForest is what makes confirmation affordable."""
    stages = artifact("data/bench_edge.json")["stages"]
    ratio = stages["isolation_forest"]["p95_us"] / stages["onnx_fp32"]["p95_us"]
    assert_in_readme(f"**{ratio:.0f}× cheaper**", "the ONNX-vs-forest ratio")


# -- INT8, rejected on measurement --------------------------------------


def test_int8_rejection_numbers_match_the_model_card():
    meta = artifact("models/edge_anomaly_meta.json")
    assert_in_readme(f"{meta['int8']['bytes'] / 1024:.1f} KB vs "
                     f"{meta['fp32']['bytes'] / 1024:.1f} KB",
                     "INT8 being larger than FP32")
    assert_in_readme(f"({meta['int8']['max_drift']:.3f})", "quantisation noise")
    assert_in_readme(f"({meta['threshold']})", "the decision threshold")
    assert_in_readme(f"**{100 * meta['int8']['false_positive_rate']:.0f} % of healthy",
                     "the INT8 false-positive rate")


# -- the risk model ------------------------------------------------------


def test_risk_auc_is_always_quoted_against_its_ceiling():
    """The bare AUC invites a fair 'barely better than guessing'. The ceiling answers it."""
    m = artifact("models/risk_meta.json")["metrics"]
    assert_in_readme(f"{m['roc_auc']:.3f}", "the ROC-AUC")
    assert_in_readme(f"{m['bayes_ceiling_roc_auc']:.3f}", "the Bayes ceiling")
    # And they must be near each other, not pages apart.
    text = README.replace("−", "-")
    auc_at = text.find(f"{m['roc_auc']:.3f}")
    ceiling_at = text.find(f"{m['bayes_ceiling_roc_auc']:.3f}")
    assert auc_at >= 0 and ceiling_at >= 0
    assert abs(auc_at - ceiling_at) < 400, \
        "the AUC and its ceiling have drifted apart in the text"


# -- the Sentinel-2 bake -------------------------------------------------


def test_ndvi_scene_provenance_matches_the_bake():
    n = artifact("data/ndvi.json")
    if n.get("source") != "sentinel-2-l2a":
        pytest.skip("only a synthetic NDVI bake is committed")
    assert_in_readme(n["scene_id"], "the Sentinel-2 scene id")
    assert_in_readme(n["sensing_date"][:10], "the sensing date")
    assert_in_readme(f"{n['cloud_cover_pct']} % cloud", "the scene cloud cover")
    assert_in_readme(f"{n['buffer_m']:.0f} m feeder-corridor buffer", "the buffer radius")


def test_the_ndvi_finding_matches_the_data():
    """The invented feature overstated vegetation risk fleet-wide. Numbers, not vibes."""
    from twinsync.encroachment import Encroachment

    n = artifact("data/ndvi.json")
    if n.get("source") != "sentinel-2-l2a":
        pytest.skip("only a synthetic NDVI bake is committed")

    observed = [v["ndvi"] for v in n["per_tower"].values()]
    observed.sort()
    median = observed[len(observed) // 2]
    assert_in_readme(f"median NDVI {median:.3f}", "the observed median NDVI")
    assert_in_readme(f"{min(observed):.3f} to {max(observed):.3f}", "the NDVI range")

    hashed = Encroachment.hashed(n["per_tower"])
    invented = sum(hashed.risk_for(k) for k in n["per_tower"]) / len(n["per_tower"])
    real = sum(v["encroachment_risk"] for v in n["per_tower"].values()) / len(n["per_tower"])
    assert_in_readme(f"**{invented:.2f}**", "the invented feature's mean pressure")
    assert_in_readme(f"**{real:.2f}**", "the observed mean pressure")


# -- the scale table -----------------------------------------------------


def test_scale_table_matches_the_benchmark():
    rows = artifact("data/bench_twin.json")["rows"]
    for row in rows:
        assert_in_readme(f"| {row['sites']} | "
                         f"{row['coverage_precompute_ms'] / 1000:.1f} s | "
                         f"{row['stdbscan']['p95_ms']:.2f} ms | "
                         f"{row['frame_ms_p95']:.1f} ms | "
                         f"{row['frame_budget_pct']:.1f} % |",
                         f"the {row['sites']}-site row of the scale table")


def test_the_linear_scaling_claim_is_arithmetic_on_the_benchmark():
    rows = artifact("data/bench_twin.json")["rows"]
    first, last = rows[0], rows[-1]
    growth = last["frame_ms_p95"] / first["frame_ms_p95"]
    factor = last["sites"] / first["sites"]
    assert_in_readme(f"{factor:.0f}× the fleet costs {growth:.1f}× the frame",
                     "the linear-scaling claim")
    assert_in_readme(f"{last['frame_budget_pct']:.1f} % of its frame budget",
                     "the frame budget at the top of the range")
