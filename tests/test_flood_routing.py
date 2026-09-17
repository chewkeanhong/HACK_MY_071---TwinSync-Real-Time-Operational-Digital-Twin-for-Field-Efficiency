"""Flood-adaptive routing: the DEM decides which roads a van can still use.

Deliberately a separate file from `test_routing.py`, which is left untouched. Those
seven tests are the regression guard for this change -- if adding a water level altered
ordinary routing at all, they would be the ones to say so.

The scene is a north-south trench. `Valley Road` runs along the bottom of it, `Slow
Street` stays up on the high ground, and the two meet at both ends. Dry, the valley is
the fast way round. Under water it must stop being the answer.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from twinsync.geo import LocalFrame
from twinsync.routing import DRY, FLOOD_SLOWDOWN, IMPASSABLE_DEPTH_M, RoadNetwork
from twinsync.terrain import Terrain, TerrainMeta

LON0, LAT0 = 101.710, 3.150
NORTH = 0.009          # ~1000 m of latitude
EAST = 0.003           # ~333 m of longitude at this latitude

TRENCH_ELEV = 30.0
HIGH_ELEV = 40.0


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


def flood_terrain() -> Terrain:
    """High ground everywhere except a north-south trench where the fast road runs.

    40 x 40 cells of 50 m from (-500, -500), so it covers the whole scene. Ground is
    40 m ASL except within 120 m of x = 333, which sits at 30 m. Twelve percent of the
    grid is trench, so the bottom-decile low-lying threshold lands exactly on 30 m.
    """
    size, cell = 40, 50.0
    xs = -500.0 + np.arange(size) * cell
    grid = np.full((size, size), HIGH_ELEV)
    grid[:, np.abs(xs - 333.0) <= 120.0] = TRENCH_ELEV
    meta = TerrainMeta("test-dem", "", False, cell, float(grid.min()), float(grid.max()))
    return Terrain(grid, -500.0, -500.0, cell, meta)


@pytest.fixture
def frame():
    return LocalFrame(LON0, LAT0)


@pytest.fixture
def valley_network(tmp_path, frame):
    """Both roads: the fast one through the trench, the slow one on high ground."""
    south, north = [LON0, LAT0], [LON0, LAT0 + NORTH]
    features = [
        road("Slow Street", "residential", [south, north]),
        road("Valley Road", "primary", [
            south, [LON0 + EAST, LAT0], [LON0 + EAST, LAT0 + NORTH], north,
        ]),
    ]
    path = write_roads(tmp_path / "roads.geojson", features)
    return RoadNetwork.load(path, frame, terrain=flood_terrain())


@pytest.fixture
def trench_only_network(tmp_path, frame):
    """One road, running north along the bottom of the trench, and no alternative."""
    coords = [[LON0 + EAST, LAT0], [LON0 + EAST, LAT0 + NORTH]]
    path = write_roads(tmp_path / "roads.geojson", [road("Valley Road", "primary", coords)])
    return RoadNetwork.load(path, frame, terrain=flood_terrain())


# -- the DEM reaches the graph -----------------------------------------


def test_edge_elevation_is_baked_from_the_dem(valley_network):
    network = valley_network
    assert network.edge_elev is not None
    assert len(network.edge_elev) == network.graph.number_of_edges()

    direct = flood_terrain().elevation_at(network.edge_midpoints[:, 0],
                                          network.edge_midpoints[:, 1])
    assert np.allclose(network.edge_elev, direct)
    for _, _, data in network.graph.edges(data=True):
        assert data["elev"] in (TRENCH_ELEV, HIGH_ELEV)


def test_flood_datum_comes_from_the_dem_not_from_zero(valley_network):
    """A water level of 1.0 m would flood nothing here -- the ground starts at 30 m.

    This is why the dashboard talks in surcharge above the drainage line and the
    physics talks in metres above sea level.
    """
    network = valley_network
    assert network.flood_source == "test-dem"
    assert network.flood_datum == pytest.approx(TRENCH_ELEV)
    assert network.surcharge_to_level(0.5) == pytest.approx(TRENCH_ELEV + 0.5)
    assert network.set_water_level(1.0) == 0


def test_attach_terrain_is_idempotent(tmp_path, frame):
    """weather.flooded_segments attaches lazily and may call it every scan."""
    coords = [[LON0, LAT0], [LON0, LAT0 + NORTH]]
    path = write_roads(tmp_path / "roads.geojson", [road("A", "primary", coords)])
    network = RoadNetwork.load(path, frame)
    assert network.edge_elev is None

    network.attach_terrain(flood_terrain())
    first = network.edge_elev.copy()
    network.attach_terrain(flood_terrain())
    assert np.array_equal(network.edge_elev, first)


# -- the cost curve ----------------------------------------------------


def test_flood_factor_shape(valley_network):
    """Dry is free, the edge of passability costs FLOOD_SLOWDOWN, beyond it is infinite."""
    network = valley_network
    assert network.flood_factor(0.0) == 1.0
    assert network.flood_factor(-3.0) == 1.0
    assert network.flood_factor(IMPASSABLE_DEPTH_M) == float("inf")
    assert network.flood_factor(IMPASSABLE_DEPTH_M + 1.0) == float("inf")

    shallow = network.flood_factor(IMPASSABLE_DEPTH_M / 2.0)
    assert 1.0 < shallow < FLOOD_SLOWDOWN
    # Square-root shape: half the depth costs more than half the penalty.
    assert shallow > 1.0 + (FLOOD_SLOWDOWN - 1.0) / 2.0


def test_flood_factor_is_vectorised(valley_network):
    depths = np.array([-1.0, 0.0, 0.25, 0.5, 2.0])
    factors = valley_network.flood_factor(depths)
    assert factors.shape == depths.shape
    assert factors[0] == 1.0 and factors[1] == 1.0
    assert np.isinf(factors[3]) and np.isinf(factors[4])


# -- routing decisions -------------------------------------------------


def test_dry_water_level_reproduces_the_ordinary_route(valley_network):
    """DRY must be a no-op: the same answer as route(), to the second."""
    network = valley_network
    start, end = np.array([0.0, 0.0]), np.array([0.0, 1000.0])
    plain = network.route(start, end)
    answer = network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), DRY, lonlat=False)

    assert answer.status == "dry"
    assert answer.route.nodes == plain.nodes
    assert answer.route.travel_time_s == pytest.approx(plain.travel_time_s)
    assert answer.flooded_edges_crossed == 0
    assert answer.deepest_water_m == 0.0
    assert answer.detour_minutes == pytest.approx(0.0)


def test_shallow_water_is_crossed_not_refused(trench_only_network):
    """Below the impassable depth the road stays in the search, just dearer."""
    network = trench_only_network
    level = TRENCH_ELEV + IMPASSABLE_DEPTH_M / 2.0
    answer = network.get_resilient_route((333.0, 0.0), (333.0, 1000.0), level,
                                         lonlat=False)

    assert answer.status == "dry", "0.25 m is passable, so no wading is required"
    assert answer.flooded_edges_crossed == 1
    assert answer.deepest_water_m == pytest.approx(IMPASSABLE_DEPTH_M / 2.0)
    assert answer.detour_minutes > 0.0, "wet tarmac still costs time"


def test_deep_water_closes_the_only_road_and_reports_wading(trench_only_network):
    """The all-routes-flooded case: say so, and say how deep, rather than an ETA."""
    network = trench_only_network
    level = TRENCH_ELEV + IMPASSABLE_DEPTH_M + 0.2
    answer = network.get_resilient_route((333.0, 0.0), (333.0, 1000.0), level,
                                         lonlat=False)

    assert answer.status == "wading"
    assert answer.passable and answer.route is not None
    assert answer.flooded_edges_crossed == 1
    assert answer.deepest_water_m == pytest.approx(IMPASSABLE_DEPTH_M + 0.2)


def test_deep_water_forces_the_detour_when_there_is_one(valley_network):
    """The headline claim: the fast road floods, so take the slow dry one."""
    network = valley_network
    dry = network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), DRY, lonlat=False)
    assert len(dry.route.nodes) == 4, "dry, the valley loop is the fast way"

    level = TRENCH_ELEV + IMPASSABLE_DEPTH_M + 0.1
    answer = network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), level, lonlat=False)

    assert answer.status == "dry", "a dry path exists, so it must be found"
    assert len(answer.route.nodes) == 2, "took Slow Street on the high ground"
    assert answer.flooded_edges_crossed == 0, "and stayed out of the water entirely"
    assert answer.detour_minutes > 0.0, "paying for it in time"


def test_unreachable_points_are_stranded_not_wading(tmp_path, frame):
    features = [
        road("Island A", "residential", [[LON0, LAT0], [LON0, LAT0 + 0.001]]),
        road("Island B", "residential",
             [[LON0 + 0.05, LAT0], [LON0 + 0.05, LAT0 + 0.001]]),
    ]
    path = write_roads(tmp_path / "roads.geojson", features)
    network = RoadNetwork.load(path, frame, terrain=flood_terrain())
    answer = network.get_resilient_route((0.0, 0.0), (5560.0, 0.0), 100.0, lonlat=False)

    assert answer.status == "stranded"
    assert answer.route is None and not answer.passable


def test_lonlat_and_metre_coordinates_agree(valley_network):
    """Same point, two conventions, one answer -- stated, never sniffed."""
    network = valley_network
    lon, lat = network.frame.to_lonlat(0.0, 1000.0)
    in_metres = network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), DRY, lonlat=False)
    in_lonlat = network.get_resilient_route((LON0, LAT0), (float(lon), float(lat)), DRY)

    assert in_metres.route.nodes == in_lonlat.route.nodes


# -- the persistent channel keeps its contracts ------------------------


def test_route_never_becomes_unreachable_however_deep_the_water(valley_network):
    """The persistent channel saturates. Dispatch must never be stranded by weather."""
    network = valley_network
    start, end = np.array([0.0, 0.0]), np.array([0.0, 1000.0])
    network.set_water_level(1000.0, graded=True)

    assert np.isfinite(network.travel_time(start, end))
    assert network.route(start, end) is not None


def test_flood_flag_is_a_python_bool_not_a_numpy_one(valley_network):
    """`np.bool_(False) is False` is False, and test_weather asserts exactly that."""
    network = valley_network
    network.set_water_level(TRENCH_ELEV + 0.3)
    for _, _, data in network.graph.edges(data=True):
        assert isinstance(data.get("flooded", False), bool)


def test_water_level_state_is_readable_and_clears(valley_network):
    network = valley_network
    assert network.water_level == DRY

    wet = network.set_water_level(network.surcharge_to_level(0.5))
    assert wet > 0
    assert network.water_level == pytest.approx(TRENCH_ELEV + 0.5)

    network.clear_congestion()
    assert network.water_level == DRY
    for _, _, data in network.graph.edges(data=True):
        assert data["flooded"] is False
        assert data["water_depth_m"] == 0.0
        assert data["time"] == pytest.approx(data["base_time"])


def test_flooding_does_not_erase_a_standing_jam(valley_network):
    """Same argument the weather module has always made, now in one place."""
    network = valley_network
    network.set_congestion(4.0, road_name="Valley Road")
    network.set_water_level(TRENCH_ELEV + 0.1)      # a mild flood, factor < 4.0

    for _, _, data in network.graph.edges(data=True):
        if data.get("name") == "Valley Road":
            assert data["congestion"] == pytest.approx(4.0), "the jam sets the floor"

    network.set_water_level(DRY)
    for _, _, data in network.graph.edges(data=True):
        if data.get("name") == "Valley Road":
            assert data["congestion"] == pytest.approx(4.0)


def test_flat_terrain_disables_flood_routing_instead_of_drowning_the_city(tmp_path, frame):
    """No DEM means we cannot say where water goes, so we decline to guess."""
    coords = [[LON0, LAT0], [LON0, LAT0 + NORTH]]
    path = write_roads(tmp_path / "roads.geojson", [road("Any Road", "primary", coords)])
    network = RoadNetwork.load(path, frame, terrain=Terrain.flat())

    assert network.flood_source == "no-dem"
    assert network.set_water_level(1000.0) == 0
    for _, _, data in network.graph.edges(data=True):
        assert not data.get("flooded", False)


# -- the shape the dashboard consumes ----------------------------------


def test_resilient_route_serialises_to_geojson(valley_network):
    network = valley_network
    answer = network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), DRY, lonlat=False)
    feature = answer.to_geojson(network.frame)

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "LineString"
    assert len(feature["geometry"]["coordinates"]) == len(answer.route.nodes)
    assert feature["properties"]["status"] == "dry"
    assert feature["properties"]["water_level"] is None      # DRY is not a number
    for lon, lat in feature["geometry"]["coordinates"]:
        assert abs(lon - LON0) < 0.1 and abs(lat - LAT0) < 0.1


def test_stranded_route_still_serialises(tmp_path, frame):
    """The client must get a well-formed Feature even when there is no path."""
    coords = [[LON0, LAT0], [LON0, LAT0 + 0.001]]
    path = write_roads(tmp_path / "roads.geojson", [road("A", "primary", coords)])
    network = RoadNetwork.load(path, frame, terrain=flood_terrain())
    answer = network.get_resilient_route((0.0, 0.0), (99999.0, 0.0), 35.0, lonlat=False)

    feature = answer.to_geojson(network.frame)
    assert feature["properties"]["status"] == "stranded"
    assert feature["geometry"]["coordinates"] == []
    assert feature["properties"]["minutes"] is None


def test_resilient_route_does_not_mutate_the_graph(valley_network):
    """It answers HTTP requests beside the running sim loop, so it must be pure."""
    network = valley_network
    before = [dict(data) for _, _, data in network.graph.edges(data=True)]
    network.get_resilient_route((0.0, 0.0), (0.0, 1000.0), 1000.0, lonlat=False)
    after = [dict(data) for _, _, data in network.graph.edges(data=True)]

    assert before == after
    assert network.water_level == DRY
