"""Vegetation encroachment, and the honesty of its provenance labelling.

This was the last input in the pipeline that was invented rather than observed. The
tests that matter here are less about arithmetic than about labelling: a simulated value
that reports itself as real is worse than no value at all, so the fallback path is
tested as carefully as the real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from twinsync.encroachment import SIMULATED_TAG, Encroachment

DATA = Path(__file__).resolve().parents[1] / "data"


# -- the fallback --------------------------------------------------------


def test_hashed_fallback_is_stable_across_processes():
    """Seeded from sha1, never `hash()`, which CPython salts per process.

    A per-process seed here would make the A/B comparison drift between runs, which has
    already cost this project a stash-and-bisect cycle once.
    """
    first = Encroachment.hashed(["KL-01", "KL-02", "KL-03"])
    second = Encroachment.hashed(["KL-01", "KL-02", "KL-03"])
    assert first.per_tower == second.per_tower
    # A literal, so a refactor that changes the derivation is caught rather than
    # silently producing a different set of numbers.
    assert first.risk_for("KL-01") == pytest.approx(0.41)


def test_hashed_fallback_never_claims_to_be_an_observation():
    fallback = Encroachment.hashed(["KL-01"])
    assert not fallback.is_real
    assert fallback.source_tag == SIMULATED_TAG
    assert "simulated" in fallback.describe().lower()
    assert fallback.ndvi_for("KL-01") is None


def test_missing_file_yields_no_object_rather_than_silent_zeros():
    assert Encroachment.load(DATA / "does-not-exist.json") is None


def test_unknown_tower_falls_back_rather_than_raising():
    """A site added to the scenario but not to the bake must not crash the run."""
    fallback = Encroachment.hashed(["KL-01"])
    assert fallback.risk_for("KL-99") == pytest.approx(0.4)


# -- the baked observation ----------------------------------------------


def baked() -> Encroachment:
    loaded = Encroachment.load(DATA / "ndvi.json")
    assert loaded is not None, "data/ndvi.json is missing -- run scripts/fetch_ndvi.py"
    return loaded


def test_baked_ndvi_covers_every_site_in_the_aoi():
    towers = json.loads((DATA / "towers.geojson").read_text(encoding="utf-8"))
    known = {f["properties"]["id"] for f in towers["features"]}
    assert known <= set(baked().per_tower), "some sites have no NDVI sample"


def test_baked_ndvi_reports_its_scene():
    """The provenance is the point: a judge must be able to look the scene up."""
    encroachment = baked()
    if not encroachment.is_real:
        pytest.skip("only a synthetic bake is committed")
    assert encroachment.meta["scene_id"]
    assert encroachment.meta["sensing_date"]
    assert encroachment.source_tag.endswith(encroachment.meta["scene_id"])
    assert "Sentinel-2" in encroachment.describe()


def test_baked_ndvi_values_are_physically_possible():
    for tower_id, entry in baked().per_tower.items():
        risk = entry["encroachment_risk"]
        assert 0.0 <= risk <= 1.0, f"{tower_id} risk {risk} out of range"
        if entry.get("ndvi") is not None:
            assert -1.0 <= entry["ndvi"] <= 1.0, f"{tower_id} NDVI out of range"


def test_baked_ndvi_is_a_dense_urban_signature():
    """The CBD is built up, so NDVI must read low almost everywhere.

    This is the assertion that would have caught a band mix-up: swapping red and NIR
    flips the sign, and a scene of downtown Kuala Lumpur reading like rainforest is the
    symptom. It also pins the finding that the hashed stand-in this replaced was
    overstating vegetation pressure across the whole fleet.
    """
    encroachment = baked()
    if not encroachment.is_real:
        pytest.skip("only a synthetic bake is committed")

    values = [e["ndvi"] for e in encroachment.per_tower.values()
              if e.get("ndvi") is not None]
    assert values
    assert max(values) < 0.6, "a CBD site should not read as dense canopy"
    assert min(values) > -0.2, "negative NDVI here would mean water or a sign flip"
    assert sorted(values)[len(values) // 2] < 0.35, "median NDVI is too green for a CBD"


def test_real_observation_is_lower_pressure_than_the_stand_in_it_replaced():
    """The invented feature was systematically overstating vegetation risk.

    Recorded as a test because it is a finding worth not losing: swapping a plausible
    guess for a measurement moved the whole fleet down, not around.
    """
    encroachment = baked()
    if not encroachment.is_real:
        pytest.skip("only a synthetic bake is committed")

    hashed = Encroachment.hashed(encroachment.per_tower)
    observed = [encroachment.risk_for(t) for t in encroachment.per_tower]
    invented = [hashed.risk_for(t) for t in encroachment.per_tower]
    assert sum(observed) / len(observed) < sum(invented) / len(invented)


# -- the join into the world --------------------------------------------


def test_world_exposes_encroachment_and_never_leaves_it_none():
    from twinsync.world import World

    world = World.load(DATA, require_towers=True)
    assert world.encroachment is not None
    for tower in world.towers:
        assert 0.0 <= world.encroachment.risk_for(tower.id) <= 1.0
