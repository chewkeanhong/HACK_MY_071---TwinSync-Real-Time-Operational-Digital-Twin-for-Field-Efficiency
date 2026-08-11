"""Geometry primitives, checked against cases computable by hand."""

from __future__ import annotations

import numpy as np
import pytest

from twinsync.geo import (
    LocalFrame,
    PolygonSet,
    haversine_m,
    point_in_ring,
    ring_area,
    ring_centroid,
    ring_perimeter,
    segment_ring_crossings,
)

from .conftest import square


def test_local_frame_round_trip():
    frame = LocalFrame(101.71, 3.15)
    lon, lat = frame.to_lonlat(*frame.to_xy(101.715, 3.155))
    assert lon == pytest.approx(101.715, abs=1e-9)
    assert lat == pytest.approx(3.155, abs=1e-9)


def test_local_frame_origin_is_zero():
    frame = LocalFrame(101.71, 3.15)
    x, y = frame.to_xy(101.71, 3.15)
    assert float(x) == pytest.approx(0.0)
    assert float(y) == pytest.approx(0.0)


def test_local_frame_matches_haversine_over_short_distances():
    """The flat-earth shortcut must not cost us more than a metre across the AOI."""
    frame = LocalFrame(101.715, 3.151)
    lon, lat = 101.725, 3.161          # ~1.5 km away, the far corner of the AOI
    x, y = frame.to_xy(lon, lat)
    planar = float(np.hypot(x, y))
    great_circle = float(haversine_m(101.715, 3.151, lon, lat))
    assert planar == pytest.approx(great_circle, rel=1e-3)


def test_ring_area_and_perimeter_of_a_square():
    ring = square(0.0, 0.0, 10.0)
    assert ring_area(ring) == pytest.approx(100.0)
    assert ring_perimeter(ring) == pytest.approx(40.0)


def test_ring_centroid_of_offset_square():
    cx, cy = ring_centroid(square(25.0, -10.0, 8.0))
    assert cx == pytest.approx(25.0)
    assert cy == pytest.approx(-10.0)


def test_point_in_ring():
    ring = square(0.0, 0.0, 10.0)
    assert point_in_ring(0.0, 0.0, ring) is True
    assert point_in_ring(4.9, 4.9, ring) is True
    assert point_in_ring(5.1, 0.0, ring) is False
    assert point_in_ring(-100.0, 0.0, ring) is False


def test_segment_crossings_straight_through():
    """A segment crossing a 10 m square centred at the origin enters at 45%, exits at 55%."""
    ring = square(0.0, 0.0, 10.0)
    crossings = segment_ring_crossings(
        np.array([-50.0, 0.0]), np.array([50.0, 0.0]), ring
    )
    assert len(crossings) == 2
    assert crossings[0] == pytest.approx(0.45)
    assert crossings[1] == pytest.approx(0.55)


def test_segment_crossings_miss_returns_nothing():
    ring = square(0.0, 0.0, 10.0)
    crossings = segment_ring_crossings(
        np.array([-50.0, 100.0]), np.array([50.0, 100.0]), ring
    )
    assert len(crossings) == 0


def test_polygon_set_indexes_and_filters_candidates():
    rings = [square(0.0, 0.0, 10.0), square(500.0, 500.0, 10.0)]
    polygons = PolygonSet(rings, np.array([20.0, 30.0]), cell_size=50.0)

    # A ray along the x axis near the origin must not consider the distant building.
    near = polygons.candidates_near_segment(np.array([-50.0, 0.0]), np.array([50.0, 0.0]))
    assert set(near.tolist()) == {0}

    both = polygons.candidates_near_segment(np.array([0.0, 0.0]), np.array([500.0, 500.0]))
    assert set(both.tolist()) == {0, 1}


def test_polygon_set_radius_query():
    rings = [square(0.0, 0.0, 10.0), square(300.0, 0.0, 10.0)]
    polygons = PolygonSet(rings, np.array([20.0, 20.0]), cell_size=50.0)

    assert set(polygons.candidates_within(np.array([0.0, 0.0]), 100.0).tolist()) == {0}
    assert set(polygons.candidates_within(np.array([0.0, 0.0]), 400.0).tolist()) == {0, 1}


def test_polygon_set_ring_accessor_round_trips():
    rings = [square(0.0, 0.0, 10.0), square(100.0, 0.0, 20.0)]
    polygons = PolygonSet(rings, np.array([10.0, 10.0]), cell_size=50.0)
    assert np.allclose(polygons.ring(0), rings[0])
    assert np.allclose(polygons.ring(1), rings[1])
