"""The assertion the entire pitch rests on.

If a tall building sits between an antenna and a target, the target is dark -- and a flat
2D range check does not know that. Everything else in TwinSync (impact scoring, dispatch
priority, the outage shadow on screen) is downstream of this being correct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from twinsync.coverage import CoverageEngine
from twinsync.world import World

from .conftest import make_world


def test_tall_building_blocks_line_of_sight(blocked_scene):
    """The headline claim: in range on a flat map, dark in three dimensions."""
    engine = CoverageEngine(blocked_scene).compute(verbose=False)
    coverage = engine.by_tower["T1"]

    # A 2D radius model sees both buildings well inside 500 m.
    assert coverage.in_radius == {"target", "blocker"}

    # In 3D the target is in the blocker's shadow.
    assert "target" not in coverage.covered
    assert "target" in coverage.blocked

    # The blocker itself is 100 m tall and directly visible from a 30 m antenna.
    assert "blocker" in coverage.covered


def test_removing_the_blocker_restores_line_of_sight(blocked_scene):
    """Same geometry minus the obstruction -- the target must come back."""
    scene = make_world(
        specs=[{"id": "target", "cx": 200.0, "cy": 0.0, "size": 20.0, "height": 20.0}],
        towers=[{"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 30.0, "range_m": 500.0}],
    )
    engine = CoverageEngine(scene).compute(verbose=False)

    assert "target" in engine.by_tower["T1"].covered
    assert engine.by_tower["T1"].blocked == set()


def test_short_obstruction_does_not_block(blocked_scene):
    """A ray passing 24 m up is not stopped by a 5 m shophouse."""
    scene = make_world(
        specs=[
            {"id": "target", "cx": 200.0, "cy": 0.0, "size": 20.0, "height": 20.0},
            {"id": "shophouse", "cx": 100.0, "cy": 0.0, "size": 40.0, "height": 5.0},
        ],
        towers=[{"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 30.0, "range_m": 500.0}],
    )
    engine = CoverageEngine(scene).compute(verbose=False)
    assert "target" in engine.by_tower["T1"].covered


def test_2d_overestimates_coverage(blocked_scene):
    """The demo statistic: the flat model claims more than reality delivers."""
    engine = CoverageEngine(blocked_scene).compute(verbose=False)
    stats = engine.compare_2d_vs_3d("T1")

    assert stats["buildings_2d"] > stats["buildings_3d"]
    assert stats["blocked"] == 1
    assert stats["subscribers_2d"] > stats["subscribers_3d"]


def test_outage_only_counts_buildings_with_no_other_tower():
    """A building served by two towers survives losing one of them."""
    scene = make_world(
        specs=[{"id": "office", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 20.0,
                "subscribers": 500}],
        towers=[
            {"id": "T1", "x": -150.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T2", "x": 150.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
        ],
    )
    engine = CoverageEngine(scene).compute(verbose=False)
    assert engine.served_by("office") == {"T1", "T2"}

    # Losing one tower leaves the other serving it.
    assert engine.outage({"T1"}) == set()
    # Losing both puts it dark.
    assert engine.outage({"T1", "T2"}) == {"office"}
    assert engine.subscribers_affected({"office"}) == 500


def test_coverage_cache_round_trip(blocked_scene, tmp_path):
    """A cached computation must restore identically, or startup is lying."""
    engine = CoverageEngine(blocked_scene).compute(verbose=False)
    cache = tmp_path / "coverage.json"
    engine.save(cache)

    restored = CoverageEngine(blocked_scene)
    assert restored.load(cache) is True
    assert restored.by_tower["T1"].covered == engine.by_tower["T1"].covered
    assert restored.by_tower["T1"].blocked == engine.by_tower["T1"].blocked
    assert restored.served_by("blocker") == engine.served_by("blocker")


def test_cache_rejected_when_towers_change(blocked_scene, tmp_path):
    """A stale cache from a different tower layout must not be silently trusted."""
    engine = CoverageEngine(blocked_scene).compute(verbose=False)
    cache = tmp_path / "coverage.json"
    engine.save(cache)

    other = make_world(
        specs=[{"id": "target", "cx": 200.0, "cy": 0.0, "size": 20.0, "height": 20.0}],
        towers=[{"id": "DIFFERENT", "x": 0.0, "y": 0.0, "antenna_height": 30.0}],
    )
    assert CoverageEngine(other).load(cache) is False


def test_2d_model_misses_buildings_that_are_genuinely_dark():
    """The claim the pitch actually rests on, pinned down.

    Two towers whose radii both reach an office, and a tower block standing between the
    office and the second tower. A 2D model sees two overlapping circles and concludes
    the office is fine when the first tower fails. In 3D it cannot see the survivor, so
    it is dark. The flat map does not overstate the outage -- it misses it.
    """
    scene = make_world(
        specs=[
            {"id": "office", "cx": 0.0, "cy": 0.0, "size": 20.0, "height": 20.0,
             "subscribers": 400},
            # Blocks the office's view of T2 only.
            {"id": "wall", "cx": 150.0, "cy": 0.0, "size": 60.0, "height": 180.0},
        ],
        towers=[
            {"id": "T1", "x": -150.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
            {"id": "T2", "x": 300.0, "y": 0.0, "antenna_height": 40.0, "range_m": 500.0},
        ],
    )
    engine = CoverageEngine(scene).compute(verbose=False)

    # Both towers have the office inside their radius, so a flat model sees redundancy.
    assert "office" in engine.by_tower["T1"].in_radius
    assert "office" in engine.by_tower["T2"].in_radius

    # But only T1 can actually see it.
    assert engine.served_by("office") == {"T1"}

    failed = {"T1"}
    assert engine.outage_2d(failed) == set(), "a 2D model should see T2 as a survivor"
    assert engine.outage(failed) == {"office"}, "in 3D the office is dark"

    # The headline: the flat model reports nobody affected while 400 people are off air.
    assert engine.subscribers_affected(engine.outage_2d(failed)) == 0
    assert engine.subscribers_affected(engine.outage(failed)) == 400


def test_naive_radius_overstates_in_the_other_direction():
    """The naive circle-count errs the opposite way, so we must not claim it as the win.

    Counting every building inside a failed tower's circle exaggerates: on the real
    scene it reports 723 buildings against a true 40. Both flat readings are wrong --
    one panics, one shrugs -- and only the shrugging one is worth putting on a slide,
    because that is the one that leaves people off the air.
    """
    scene = make_world(
        specs=[
            {"id": "near", "cx": 60.0, "cy": 0.0, "size": 20.0, "height": 15.0},
            {"id": "wall", "cx": 160.0, "cy": 0.0, "size": 70.0, "height": 200.0},
            {"id": "far", "cx": 300.0, "cy": 0.0, "size": 20.0, "height": 15.0},
            {"id": "further", "cx": 360.0, "cy": 0.0, "size": 20.0, "height": 15.0},
        ],
        towers=[{"id": "T1", "x": -60.0, "y": 0.0, "antenna_height": 45.0,
                 "range_m": 600.0}],
    )
    engine = CoverageEngine(scene).compute(verbose=False)

    naive = engine.naive_radius({"T1"})
    real = engine.outage({"T1"})

    assert len(naive) == 4                     # everything sits inside the circle
    assert {"far", "further"} <= naive         # ...including the two in the shadow
    assert {"far", "further"} & real == set()  # which never had service to lose
    assert len(naive) > len(real)


# ---------------------------------------------------------------- the real scene


@pytest.fixture(scope="module")
def kl_coverage():
    """The committed KL CBD scene, computed once for the whole module.

    Uses the on-disk cache when its fingerprint matches; otherwise recomputes, which
    takes about 25 s. That is worth paying once to keep the pitch's headline number
    honest.
    """
    data = Path(__file__).resolve().parents[1] / "data"
    if not (data / "towers.geojson").exists():
        pytest.skip("no committed scene")

    world = World.load(data, require_towers=True)
    engine = CoverageEngine(world)
    if not engine.load(data / "coverage_cache.json"):
        engine.compute(verbose=False)
    return engine


SCENARIO_FAULTS = {"KL-03", "KL-13", "KL-09", "KL-06"}


def test_headline_number_on_the_real_scene(kl_coverage):
    """The figure the pitch is built on, pinned to the committed data.

    README quotes these directly. If a change to the DEM, the Fresnel criterion or the
    height imputation moves them, this fails and the README has to be updated with it --
    which is the entire point. Do not widen the tolerance to make a red test green.
    """
    dark_3d = kl_coverage.outage(SCENARIO_FAULTS)
    dark_2d = kl_coverage.outage_2d(SCENARIO_FAULTS)

    assert len(dark_3d) == 47
    assert kl_coverage.subscribers_affected(dark_3d) == 9026
    assert len(dark_2d) == 3
    assert kl_coverage.subscribers_affected(dark_2d) == 3

    # The claim in one line: the flat map misses almost all of them.
    assert kl_coverage.subscribers_affected(dark_3d) - \
           kl_coverage.subscribers_affected(dark_2d) == 9023


def test_naive_circle_count_is_the_strawman_we_do_not_quote(kl_coverage):
    """Also pinned, because the README explicitly promises *not* to use this number."""
    assert len(kl_coverage.naive_radius(SCENARIO_FAULTS)) == 1054


def test_every_scenario_tower_is_worse_in_3d_than_a_flat_model_thinks(kl_coverage):
    """Direction matters more than the exact count: 2D must never overstate the outage."""
    for tower_id in sorted(SCENARIO_FAULTS):
        failed = {tower_id}
        assert (kl_coverage.subscribers_affected(kl_coverage.outage(failed))
                >= kl_coverage.subscribers_affected(kl_coverage.outage_2d(failed))), \
            f"{tower_id}: the flat model claimed more outage than 3D found"


def test_terrain_backing_the_scene_is_real_copernicus_data(kl_coverage):
    """Guards the GeoAI claim at its source."""
    assert kl_coverage.world.terrain.meta.source == "copernicus-glo30"
    assert kl_coverage.world.towers[0].antenna_z > kl_coverage.world.towers[0].antenna_height
