"""Helpers for building small, hand-checkable scenes.

The real scene has 2000 buildings and nobody can reason about it. Every geometric claim
in these tests is made against a world small enough to verify on paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from twinsync.geo import LocalFrame, PolygonSet, ring_centroid
from twinsync.world import Building, Tower, World


def square(cx: float, cy: float, size: float) -> np.ndarray:
    """A closed square ring centred on (cx, cy), in local metres."""
    h = size / 2.0
    return np.array([
        [cx - h, cy - h], [cx + h, cy - h], [cx + h, cy + h],
        [cx - h, cy + h], [cx - h, cy - h],
    ], dtype=np.float64)


def make_world(specs: list[dict], towers: list[dict] | None = None) -> World:
    """Build a World from square footprints.

    specs: [{"id", "cx", "cy", "size", "height", "subscribers"?, "critical"?}, ...]
    towers: [{"id", "x", "y", "antenna_height", "range_m", "host"?}, ...]
    """
    frame = LocalFrame(101.71, 3.15)
    rings, buildings, heights = [], [], []

    for index, spec in enumerate(specs):
        ring = square(spec["cx"], spec["cy"], spec["size"])
        centroid = np.array(ring_centroid(ring))
        lon, lat = frame.to_lonlat(centroid[0], centroid[1])
        buildings.append(Building(
            id=spec["id"],
            index=index,
            name=spec.get("name"),
            kind=spec.get("kind", "office"),
            height=float(spec["height"]),
            height_source=spec.get("height_source", "osm"),
            area_m2=float(spec["size"] ** 2),
            perimeter_m=float(4 * spec["size"]),
            centroid_xy=centroid,
            centroid_lonlat=(float(lon), float(lat)),
            subscribers=int(spec.get("subscribers", 100)),
            critical=bool(spec.get("critical", False)),
            ring_lonlat=[],
        ))
        rings.append(ring)
        heights.append(float(spec["height"]))

    polygons = PolygonSet(rings, np.array(heights), cell_size=50.0)

    tower_objects = []
    for spec in towers or []:
        xy = np.array([float(spec["x"]), float(spec["y"])])
        lon, lat = frame.to_lonlat(xy[0], xy[1])
        tower_objects.append(Tower(
            id=spec["id"],
            name=spec.get("name", spec["id"]),
            lon=float(lon), lat=float(lat),
            xy=xy,
            antenna_height=float(spec["antenna_height"]),
            range_m=float(spec.get("range_m", 600.0)),
            host_building=spec.get("host"),
        ))

    return World(frame, buildings, polygons, tower_objects)


@pytest.fixture
def blocked_scene() -> World:
    """Tower -- tall blocker -- short target, all on the x axis.

        tower(x=0, antenna 30m)   blocker(x=100, 100m tall)   target(x=200, 20m tall)

    The ray to the target's roof passes x=100 at ~24 m, far below the 100 m blocker,
    so the target must be dark.
    """
    return make_world(
        specs=[
            {"id": "target", "cx": 200.0, "cy": 0.0, "size": 20.0, "height": 20.0},
            {"id": "blocker", "cx": 100.0, "cy": 0.0, "size": 40.0, "height": 100.0},
        ],
        towers=[{"id": "T1", "x": 0.0, "y": 0.0, "antenna_height": 30.0, "range_m": 500.0}],
    )
