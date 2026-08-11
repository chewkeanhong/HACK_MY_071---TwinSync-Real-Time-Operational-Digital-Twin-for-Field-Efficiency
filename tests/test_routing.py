"""Routing on a hand-built street graph small enough to verify on paper.

The claim under test is the one the demo makes out loud: dispatch on travel time, not
distance, and the answer changes. If A* here ever starts preferring the short slow road,
the pitch is wrong.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from twinsync.geo import LocalFrame
from twinsync.routing import RoadNetwork

# Origin of the local frame used by every test here.
LON0, LAT0 = 101.710, 3.150

# 0.009 deg of latitude is ~1000 m; 0.003 deg of longitude is ~333 m at this latitude.
NORTH = 0.009
EAST = 0.003


def write_roads(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}),
                    encoding="utf-8")
    return path


def road(name, highway, coords, oneway="no"):
    return {
        "type": "Feature",
        "properties": {"name": name, "highway": highway, "oneway": oneway,
                       "maxspeed": None},
        "geometry": {"type": "LineString", "coordinates": coords},
    }


@pytest.fixture
def frame():
    return LocalFrame(LON0, LAT0)


@pytest.fixture
def two_route_network(tmp_path, frame):
    r"""A short slow road and a long fast road between the same two points.

        depot (0,0) ---- "Slow Street", residential, 1000 m ---- site (0, 1000)
              \                                                    /
               "Fast Loop", primary, 333 + 1000 + 333 = 1666 m ----

    Residential at 22 km/h takes ~164 s; primary at 45 km/h takes ~133 s. The long way
    round is the fast way.
    """
    south, north = [LON0, LAT0], [LON0, LAT0 + NORTH]
    features = [
        road("Slow Street", "residential", [south, north]),
        road("Fast Loop", "primary", [
            south,
            [LON0 + EAST, LAT0],
            [LON0 + EAST, LAT0 + NORTH],
            north,
        ]),
    ]
    return RoadNetwork.load(write_roads(tmp_path / "roads.geojson", features), frame)


def test_graph_is_built_and_connected(two_route_network):
    network = two_route_network
    assert network.graph.number_of_nodes() == 4
    # Both roads are two-way, so every edge appears in both directions.
    assert network.graph.number_of_edges() == 8


def test_prefers_the_longer_but_faster_road(two_route_network):
    """The headline routing claim."""
    network = two_route_network
    start = np.array([0.0, 0.0])
    end = network.node_xy[network.nearest_node(np.array([0.0, 1000.0]))]

    route = network.route(start, end)
    assert route is not None

    # It picked the loop: four nodes, not the two-node direct road.
    assert len(route.nodes) == 4
    assert route.distance_m > 1600.0          # the long way round
    assert route.travel_time_s < 164.0        # but quicker than the direct road

    # Sanity-check the alternative really is slower.
    direct_seconds = 1000.8 / (22 / 3.6) + 4.0
    assert route.travel_time_s < direct_seconds


def test_congestion_flips_the_decision(two_route_network):
    """Jam the fast road and the short road wins -- this is what the demo triggers."""
    network = two_route_network
    start = np.array([0.0, 0.0])
    end = network.node_xy[network.nearest_node(np.array([0.0, 1000.0]))]

    before = network.route(start, end)
    assert len(before.nodes) == 4

    affected = network.set_congestion(4.0, road_name="Fast Loop")
    assert affected > 0

    after = network.route(start, end)
    assert len(after.nodes) == 2                       # now the direct road
    assert after.travel_time_s < before.travel_time_s * 4

    network.clear_congestion()
    assert len(network.route(start, end).nodes) == 4   # and back again


def test_oneway_is_respected(tmp_path, frame):
    """A one-way street must not be driven backwards."""
    south, north = [LON0, LAT0], [LON0, LAT0 + NORTH]
    features = [road("One Way Street", "primary", [south, north], oneway="yes")]
    network = RoadNetwork.load(write_roads(tmp_path / "roads.geojson", features), frame)

    # Northbound works.
    assert network.route(np.array([0.0, 0.0]), np.array([0.0, 1000.0])) is not None
    # Southbound has no legal path.
    assert network.route(np.array([0.0, 1000.0]), np.array([0.0, 0.0])) is None


def test_travel_time_is_infinite_when_unreachable(tmp_path, frame):
    """Two disconnected roads must not produce a bogus finite ETA."""
    features = [
        road("Island A", "residential", [[LON0, LAT0], [LON0, LAT0 + 0.001]]),
        road("Island B", "residential", [[LON0 + 0.05, LAT0], [LON0 + 0.05, LAT0 + 0.001]]),
    ]
    network = RoadNetwork.load(write_roads(tmp_path / "roads.geojson", features), frame)
    assert network.travel_time(np.array([0.0, 0.0]), np.array([5560.0, 0.0])) == float("inf")


def test_posted_speed_limit_overrides_class_default(tmp_path, frame):
    """A 20 km/h posted limit on a primary road must slow the route down."""
    coords = [[LON0, LAT0], [LON0, LAT0 + NORTH]]
    fast = road("Default Primary", "primary", coords)
    slow = road("Posted Slow", "primary", coords)
    slow["properties"]["maxspeed"] = "20"

    net_fast = RoadNetwork.load(write_roads(tmp_path / "a.geojson", [fast]), frame)
    net_slow = RoadNetwork.load(write_roads(tmp_path / "b.geojson", [slow]), frame)

    start, end = np.array([0.0, 0.0]), np.array([0.0, 1000.0])
    assert net_slow.travel_time(start, end) > net_fast.travel_time(start, end)
