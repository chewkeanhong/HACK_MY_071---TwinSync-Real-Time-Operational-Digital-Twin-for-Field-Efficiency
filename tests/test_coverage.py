"""The assertion the entire pitch rests on.

If a tall building sits between an antenna and a target, the target is dark -- and a flat
2D range check does not know that. Everything else in TwinSync (impact scoring, dispatch
priority, the outage shadow on screen) is downstream of this being correct.
"""

from __future__ import annotations

from twinsync.coverage import CoverageEngine

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
