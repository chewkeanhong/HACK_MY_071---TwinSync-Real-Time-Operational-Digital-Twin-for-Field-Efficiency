"""Road network and travel-time routing.

Dispatch decisions are made on *minutes*, not metres. That distinction is the whole
argument for routing over a real street graph: the crew that is closest in a straight
line is regularly not the crew that arrives first, because of one-way systems, river
crossings and congestion. A dispatcher staring at a map with radial distance rings cannot
see that; the twin can.

Edges carry a congestion multiplier the simulation can raise at runtime, so a jam can be
introduced mid-demo and the routing answer changes with it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .geo import LocalFrame
from .terrain import Terrain
from .weather import FLOOD_SLOWDOWN

# Free-flow speeds in km/h for a dense Asian CBD -- deliberately conservative; these are
# service vans in traffic, not cars on an empty motorway.
SPEED_KMH = {
    "motorway": 70, "trunk": 55, "primary": 45, "secondary": 38, "tertiary": 32,
    "unclassified": 28, "residential": 22, "living_street": 12, "service": 14,
    "motorway_link": 45, "trunk_link": 38, "primary_link": 32,
    "secondary_link": 28, "tertiary_link": 25,
}
DEFAULT_SPEED_KMH = 25.0

# Junctions cost time even when the road is clear.
JUNCTION_PENALTY_S = 4.0

# Coordinate quantisation, in metres, used to merge shared way endpoints into one node.
NODE_SNAP_M = 0.5

# Furthest a point may be from the routable network before we call it unreachable. Without
# this, snapping to the routable core would silently teleport a genuinely stranded point
# across the city to the nearest driveable road.
MAX_SNAP_M = 250.0

# -- flooding -------------------------------------------------------------
#
# The DEM says where water collects; ``water_level`` says how much of it there is. A
# road is under ``water_level - elevation`` metres of water, and that number decides
# both how slow it is and whether a van can use it at all.

# Depth at which a service van stops being a vehicle and starts being a boat. Vehicle
# stability studies (DEFRA FD2320, Australian Rainfall & Runoff Book 6) put loss of
# control for a light van between 0.4 and 0.6 m of still water; 0.5 m is the middle of
# that band and the figure flood-response plans generally quote.
IMPASSABLE_DEPTH_M = 0.50

# What an impassable road costs when there is genuinely no dry alternative. Finite on
# purpose: a crew ordered through standing water is a decision someone can take, and
# returning infinity here would instead report the site as unreachable.
FLOOD_IMPASSABLE_FACTOR = 40.0

# Elevation spread below which we conclude the DEM carries no flood signal at all --
# i.e. Terrain.flat(), the fallback when nothing has been baked. Flooding an entire
# city because the elevation data is missing is a worse failure than declining to.
FLAT_DEM_EPS = 0.5

# The water level meaning "no standing water anywhere".
DRY = -math.inf


@dataclass
class Route:
    """A driveable path with the numbers dispatch actually cares about."""

    nodes: list[int]
    xy: np.ndarray
    distance_m: float
    travel_time_s: float

    @property
    def minutes(self) -> float:
        return self.travel_time_s / 60.0

    def to_lonlat(self, frame: LocalFrame) -> list[list[float]]:
        lon, lat = frame.to_lonlat(self.xy[:, 0], self.xy[:, 1])
        return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lon, lat)]


@dataclass(frozen=True)
class ResilientRoute:
    """A route answered against a stated water level, and how wet it is.

    ``status`` is the honest verdict of three: ``"dry"`` means a path exists that never
    enters water deeper than :data:`IMPASSABLE_DEPTH_M`; ``"wading"`` means one does not,
    and the path returned crosses water a van should not be in; ``"stranded"`` means
    there is no path at all. A dispatcher who is told "wading" can decide to send the
    crew anyway -- what they must never get is a silent ETA that quietly drove through
    a metre of floodwater.
    """

    route: Route | None
    water_level: float
    status: str
    flooded_edges_crossed: int
    deepest_water_m: float
    detour_minutes: float

    @property
    def passable(self) -> bool:
        return self.route is not None

    def to_geojson(self, frame: LocalFrame) -> dict:
        """A GeoJSON Feature the dashboard can hand straight to a deck.gl layer."""
        coordinates = self.route.to_lonlat(frame) if self.route else []
        return {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "status": self.status,
                "water_level": (None if self.water_level == DRY
                                else round(self.water_level, 2)),
                "flooded_edges_crossed": self.flooded_edges_crossed,
                "deepest_water_m": round(self.deepest_water_m, 2),
                "detour_minutes": round(self.detour_minutes, 2),
                "minutes": round(self.route.minutes, 2) if self.route else None,
                "distance_m": round(self.route.distance_m, 1) if self.route else None,
            },
        }


class RoadNetwork:
    """A directed street graph weighted by travel time."""

    def __init__(self, graph: nx.DiGraph, node_xy: np.ndarray, frame: LocalFrame,
                 terrain: "Terrain | None" = None):
        self.graph = graph
        self.node_xy = node_xy
        self.frame = frame
        self.routable = self._find_routable_core()

        # Captured once, in a stable order, so the elevation array and the edge
        # attribute dictionaries stay index-aligned. Nothing adds edges after load.
        self._edges = list(graph.edges(data=True))
        self.edge_elev: np.ndarray | None = None
        self.edge_midpoints: np.ndarray | None = None
        self.water_level: float = DRY
        self.flood_datum: float = 0.0
        self.flood_source: str = "no-dem"
        if terrain is not None:
            self.attach_terrain(terrain)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, frame: LocalFrame, *,
             terrain: "Terrain | None" = None) -> "RoadNetwork":
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        coords: dict[tuple[int, int], int] = {}
        positions: list[tuple[float, float]] = []
        graph = nx.DiGraph()

        def node_for(x: float, y: float) -> int:
            key = (int(round(x / NODE_SNAP_M)), int(round(y / NODE_SNAP_M)))
            index = coords.get(key)
            if index is None:
                index = len(positions)
                coords[key] = index
                positions.append((x, y))
                graph.add_node(index)
            return index

        for feature in data["features"]:
            props = feature["properties"]
            line = np.asarray(feature["geometry"]["coordinates"], dtype=np.float64)
            if len(line) < 2:
                continue

            x, y = frame.to_xy(line[:, 0], line[:, 1])
            speed = cls._speed_for(props)
            oneway = str(props.get("oneway", "no")).lower() in {"yes", "true", "1", "-1"}
            reversed_way = str(props.get("oneway", "")).strip() == "-1"

            indices = [node_for(float(px), float(py)) for px, py in zip(x, y)]
            for a, b in zip(indices[:-1], indices[1:]):
                if a == b:
                    continue
                length = float(np.hypot(positions[b][0] - positions[a][0],
                                        positions[b][1] - positions[a][1]))
                if length <= 0.0:
                    continue
                seconds = length / (speed / 3.6) + JUNCTION_PENALTY_S

                forward, backward = (b, a) if reversed_way else (a, b)
                graph.add_edge(forward, backward, length=length, base_time=seconds,
                               time=seconds, congestion=1.0,
                               highway=props.get("highway"), name=props.get("name"))
                if not oneway:
                    graph.add_edge(backward, forward, length=length, base_time=seconds,
                                   time=seconds, congestion=1.0,
                                   highway=props.get("highway"), name=props.get("name"))

        node_xy = np.array(positions, dtype=np.float64)
        return cls(graph, node_xy, frame, terrain)

    def attach_terrain(self, terrain: "Terrain | None") -> None:
        """Bake a ground elevation onto every edge. Idempotent, and cheap.

        Sampled at the edge midpoint and nowhere else. That is a choice, not a shortcut:
        the DEM cell is 30 m and the median road edge here is under 10 m, so one sample
        already over-resolves the grid -- and it is the same point
        :meth:`twinsync.weather.WeatherField.flooded_segments` has always sampled, which
        is what lets the water-level model reproduce the old flooding exactly.

        Both forms are kept. The array reprices seventeen thousand edges in one
        vectorised pass; the per-edge ``elev`` attribute is what the A* weight callable
        reads, where a numpy lookup per relaxation would cost more than it saves.
        """
        if terrain is None or self.edge_elev is not None:
            return
        if not self._edges:
            self.edge_midpoints = np.zeros((0, 2), dtype=np.float64)
            self.edge_elev = np.zeros(0, dtype=np.float64)
            return

        uv = np.array([(a, b) for a, b, _ in self._edges], dtype=np.int64)
        midpoints = (self.node_xy[uv[:, 0]] + self.node_xy[uv[:, 1]]) / 2.0
        elevation = np.atleast_1d(
            np.asarray(terrain.elevation_at(midpoints[:, 0], midpoints[:, 1]),
                       dtype=np.float64))
        self.edge_midpoints = midpoints
        self.edge_elev = elevation
        for (_, _, data), height in zip(self._edges, elevation):
            data["elev"] = float(height)

        spread = float(elevation.max() - elevation.min())
        if spread < FLAT_DEM_EPS:
            # Terrain.flat(), or a DEM so level it cannot tell us where water goes.
            self.flood_source = "no-dem"
            self.flood_datum = float(elevation.min())
        else:
            self.flood_source = terrain.meta.source
            self.flood_datum = float(terrain.low_lying_threshold)

    def _find_routable_core(self) -> np.ndarray:
        """Node indices in the largest strongly connected component.

        OSM extracts always contain stubs: service roads clipped by the bounding box,
        car-park spurs, one-way slips whose partner lies outside the AOI. Snapping a
        tower or a depot onto one of those makes it unreachable, and dispatch then
        reports "no crew can get there" for a site with a main road outside its door.
        Restricting the snap to the strongly connected core guarantees that any two
        points we snap are mutually driveable -- *strongly*, not weakly, because a
        one-way pair can be weakly connected and still not drivable in both directions.
        """
        if self.graph.number_of_nodes() == 0:
            return np.zeros(0, dtype=np.int64)
        largest = max(nx.strongly_connected_components(self.graph), key=len)
        if len(largest) < 2:
            # Degenerate graph (a lone one-way street, a synthetic test fixture). There
            # is no meaningful core to snap to, so do not restrict anything.
            return np.zeros(0, dtype=np.int64)
        return np.array(sorted(largest), dtype=np.int64)

    @staticmethod
    def _speed_for(props: dict) -> float:
        """Posted speed if OSM has one, otherwise a class-based default."""
        raw = props.get("maxspeed")
        if raw:
            try:
                text = str(raw).lower().replace("km/h", "").strip()
                if "mph" in text:
                    return float(text.replace("mph", "").strip()) * 1.609
                value = float(text)
                if 5.0 <= value <= 130.0:
                    return value
            except ValueError:
                pass
        return SPEED_KMH.get(props.get("highway"), DEFAULT_SPEED_KMH)

    # -- congestion ------------------------------------------------------

    def set_congestion(self, factor: float, *, road_name: str | None = None,
                       highway: str | None = None, near: np.ndarray | None = None,
                       radius_m: float = 300.0) -> int:
        """Slow down matching edges by ``factor``. Returns how many were affected.

        Used by the scenario to create the situation where the nearest crew is not the
        fastest one.
        """
        affected = 0
        for a, b, data in self.graph.edges(data=True):
            if road_name and data.get("name") != road_name:
                continue
            if highway and data.get("highway") != highway:
                continue
            if near is not None:
                midpoint = (self.node_xy[a] + self.node_xy[b]) / 2.0
                if float(np.hypot(*(midpoint - near))) > radius_m:
                    continue
            data["congestion"] = factor
            # Remembered separately so a transient penalty applied on top -- flooding,
            # which comes and goes with the storm cell -- can be lifted without also
            # erasing the scenario's standing rush-hour jam.
            data["scenario_congestion"] = factor
            data["time"] = data["base_time"] * factor
            affected += 1
        return affected

    def clear_congestion(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            data["congestion"] = 1.0
            data["scenario_congestion"] = 1.0
            data["flooded"] = False
            data["water_depth_m"] = 0.0
            data["time"] = data["base_time"]
        self.water_level = DRY

    # -- flooding --------------------------------------------------------

    def surcharge_to_level(self, surcharge_m: float) -> float:
        """Turn "N metres of water above the drainage line" into an absolute level.

        The dashboard wants a slider that reads 0 to 3 metres. The physics wants an
        altitude above sea level, because that is the datum the DEM is in. This is the
        conversion between them, and it is the reason a naive ``water_level = 1.0``
        does not silently flood nothing: this AOI sits between 22 and 66 m ASL, so one
        metre absolute is far below the entire city.
        """
        return self.flood_datum + float(surcharge_m)

    def flood_factor(self, depth_m: np.ndarray | float) -> np.ndarray | float:
        """Travel-time multiplier for water ``depth_m`` deep. Vectorised, scalar-safe.

        One at zero depth, rising to :data:`FLOOD_SLOWDOWN` as the water approaches
        :data:`IMPASSABLE_DEPTH_M`, infinite beyond it. The square-root shape means
        shallow water hurts immediately -- traffic crawls the moment the carriageway is
        wet, it does not wait until the water is deep -- which is what the flat 3.2x
        multiplier this replaces was crudely approximating.
        """
        depth = np.asarray(depth_m, dtype=np.float64)
        ratio = np.clip(depth / IMPASSABLE_DEPTH_M, 0.0, 1.0)
        factor = 1.0 + (FLOOD_SLOWDOWN - 1.0) * np.sqrt(ratio)
        factor = np.where(depth <= 0.0, 1.0, factor)
        factor = np.where(depth >= IMPASSABLE_DEPTH_M, np.inf, factor)
        return float(factor) if factor.ndim == 0 else factor

    def set_water_level(self, level: float, *, rain_mask: np.ndarray | None = None,
                        graded: bool = False) -> int:
        """Reprice every edge for a standing water level. Returns how many are wet.

        The single writer of flood state in the codebase. Rain does not reach edge
        attributes any more; :meth:`twinsync.weather.WeatherField.flooded_segments`
        computes a level and a spatial mask and delegates here, so there is one place
        where "how deep is this road" turns into "how slow is this road".

        ``graded`` is the difference between the two callers. The simulation leaves it
        off and gets the flat :data:`FLOOD_SLOWDOWN` the dispatch A/B has always been
        measured against; the operator flood control turns it on and gets the depth
        curve. Either way this channel saturates at :data:`FLOOD_IMPASSABLE_FACTOR` and
        never writes an infinite time, so :meth:`route` keeps its contract that a
        connected pair of points is always routable -- the hard mask belongs to
        :meth:`get_resilient_route`, which does not mutate anything.
        """
        if self.edge_elev is None or self.flood_source == "no-dem":
            return 0

        self.water_level = float(level)
        depth = float(level) - self.edge_elev
        if rain_mask is not None:
            # Dry ground the rain has not reached is not flooded, however low it sits.
            depth = np.where(np.asarray(rain_mask, dtype=bool), depth, -np.inf)

        # At-or-above, not strictly above: a road exactly at the water line is awash.
        # This also keeps the boundary identical to :meth:`Terrain.is_low_lying`, which
        # matters more than it looks -- 180 edges of the committed DEM sit at exactly
        # the low-lying threshold, so a strict comparison here would quietly reprice
        # 180 roads and move the A/B figures the README is held to.
        wet = depth >= 0.0
        if graded:
            factor = self.flood_factor(depth)
            factor = np.where(np.isfinite(factor), factor, FLOOD_IMPASSABLE_FACTOR)
        else:
            factor = np.where(wet, FLOOD_SLOWDOWN, 1.0)

        affected = 0
        for (_, _, data), is_wet, multiplier, metres in zip(
                self._edges, wet, factor, depth):
            standing = data.get("scenario_congestion", 1.0)
            if is_wet:
                # Take the worse of the two rather than multiplying them. A jammed road
                # that also floods is not 14x slower than free-flowing; traffic is
                # already crawling and the water sets the floor.
                realised = max(standing, float(multiplier))
                data["congestion"] = realised
                data["time"] = data["base_time"] * realised
                # A Python bool, not a numpy one: tests assert `is False` on this.
                data["flooded"] = True
                data["water_depth_m"] = float(metres)
                affected += 1
            elif data.get("flooded"):
                # Waters recede: back to whatever standing congestion the scenario set.
                data["flooded"] = False
                data["water_depth_m"] = 0.0
                data["congestion"] = standing
                data["time"] = data["base_time"] * standing
        return affected

    # -- routing ---------------------------------------------------------

    def nearest_node(self, xy: np.ndarray) -> int | None:
        """Closest driveable node, or None if the point is nowhere near the network."""
        candidates = self.routable if len(self.routable) else np.arange(len(self.node_xy))
        if not len(candidates):
            return None
        points = self.node_xy[candidates]
        distances = np.hypot(points[:, 0] - xy[0], points[:, 1] - xy[1])
        best = int(np.argmin(distances))
        if distances[best] > MAX_SNAP_M:
            return None
        return int(candidates[best])

    def _heuristic(self, a: int, b: int) -> float:
        """Straight-line time at the fastest speed in the network -- admissible, so A*
        still returns the true optimum."""
        distance = float(np.hypot(*(self.node_xy[a] - self.node_xy[b])))
        return distance / (max(SPEED_KMH.values()) / 3.6)

    def route(self, start_xy: np.ndarray, end_xy: np.ndarray) -> Route | None:
        """Fastest path between two arbitrary points, or None if unreachable."""
        start = self.nearest_node(np.asarray(start_xy, dtype=np.float64))
        end = self.nearest_node(np.asarray(end_xy, dtype=np.float64))
        if start is None or end is None:
            return None
        if start == end:
            return Route([start], self.node_xy[[start]], 0.0, 0.0)

        try:
            path = nx.astar_path(self.graph, start, end,
                                 heuristic=self._heuristic, weight="time")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        distance = sum(self.graph[a][b]["length"] for a, b in zip(path[:-1], path[1:]))
        seconds = sum(self.graph[a][b]["time"] for a, b in zip(path[:-1], path[1:]))
        return Route(path, self.node_xy[path], distance, seconds)

    def travel_time(self, start_xy: np.ndarray, end_xy: np.ndarray) -> float:
        """Seconds to drive between two points; infinite if there is no route."""
        route = self.route(start_xy, end_xy)
        return route.travel_time_s if route else math.inf

    # -- flood-adaptive routing ------------------------------------------

    def _as_xy(self, coords, lonlat: bool) -> np.ndarray:
        """Accept lon/lat or local metres, explicitly -- never by guessing."""
        point = np.asarray(coords, dtype=np.float64)
        if not lonlat:
            return point
        x, y = self.frame.to_xy(float(point[0]), float(point[1]))
        return np.array([float(x), float(y)], dtype=np.float64)

    def _dry_cost(self, data: dict) -> float:
        """Travel time with jams but without water.

        Read from ``base_time`` and the *scenario* congestion rather than from
        ``time``, because ``time`` may already carry a flood penalty this network was
        repriced with. Multiplying our own factor on top of that would count the same
        water twice.
        """
        return data["base_time"] * data.get("scenario_congestion", 1.0)

    def _flood_weight(self, level: float, *, hard: bool):
        """A* edge weight at a stated water level.

        Returning ``None`` from a networkx weight callable removes the edge from the
        search, which is how the hard mask is applied without touching the graph. That
        matters: this runs on request, concurrently with the simulation loop that owns
        the persistent water level, so a set-then-restore would race.
        """
        def weight(u: int, v: int, data: dict) -> float | None:
            cost = self._dry_cost(data)
            elevation = data.get("elev")
            if elevation is None:
                return cost
            depth = level - elevation
            if depth <= 0.0:
                return cost
            factor = self.flood_factor(depth)
            if math.isfinite(factor):
                return cost * factor
            return None if hard else cost * FLOOD_IMPASSABLE_FACTOR
        return weight

    def _measure(self, path: list[int], level: float) -> tuple[Route, int, float]:
        """Cost a chosen path at a water level, and report how wet it got."""
        distance = 0.0
        seconds = 0.0
        crossed = 0
        deepest = 0.0
        for a, b in zip(path[:-1], path[1:]):
            data = self.graph[a][b]
            distance += data["length"]
            cost = self._dry_cost(data)
            elevation = data.get("elev")
            depth = (level - elevation) if elevation is not None else -math.inf
            if depth > 0.0:
                crossed += 1
                deepest = max(deepest, float(depth))
                factor = self.flood_factor(depth)
                cost *= FLOOD_IMPASSABLE_FACTOR if not math.isfinite(factor) else factor
            seconds += cost
        return Route(path, self.node_xy[path], distance, seconds), crossed, deepest

    def get_resilient_route(self, start_coords, target_coords,
                            current_water_level: float,
                            *, lonlat: bool = True) -> ResilientRoute:
        """The safest driveable path at a given water level, avoiding flooded roads.

        Coordinates are lon/lat by default. The parameter is named ``coords`` and this
        is the entry point the dashboard calls, so it speaks the dashboard's units;
        pass ``lonlat=False`` for the local metres every other method here takes. The
        convention is stated rather than sniffed, because a longitude of 101.7 and an
        x-offset of 101.7 m are indistinguishable and guessing wrong puts a crew in the
        wrong hemisphere.

        ``current_water_level`` is metres above sea level -- see
        :meth:`surcharge_to_level` to get there from a slider reading depth.

        Four outcomes, in the order they are tried. A point too far from any road is
        ``"stranded"`` before A* runs at all. Otherwise the search first refuses every
        edge under more than :data:`IMPASSABLE_DEPTH_M` of water, giving ``"dry"``. If
        that disconnects the city -- the all-routes-flooded case -- it retries with
        those edges merely expensive rather than forbidden, giving ``"wading"`` plus the
        depth the crew would have to drive through, which is a decision a human should
        make with the number in front of them. Only if that also fails is the answer
        ``"stranded"``.

        A start point already under water does not block departure. You cannot refuse to
        let a crew leave a depot that is flooding; you can only tell them what is ahead.
        """
        level = float(current_water_level)
        start = self.nearest_node(self._as_xy(start_coords, lonlat))
        end = self.nearest_node(self._as_xy(target_coords, lonlat))
        if start is None or end is None:
            return ResilientRoute(None, level, "stranded", 0, 0.0, 0.0)
        if start == end:
            here = Route([start], self.node_xy[[start]], 0.0, 0.0)
            return ResilientRoute(here, level, "dry", 0, 0.0, 0.0)

        route: Route | None = None
        crossed, deepest, status = 0, 0.0, "stranded"
        for hard in (True, False):
            try:
                path = nx.astar_path(self.graph, start, end, heuristic=self._heuristic,
                                     weight=self._flood_weight(level, hard=hard))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            route, crossed, deepest = self._measure(path, level)
            status = "dry" if hard else "wading"
            break

        if route is None:
            return ResilientRoute(None, level, "stranded", 0, 0.0, 0.0)

        detour = 0.0
        try:
            dry_path = nx.astar_path(self.graph, start, end, heuristic=self._heuristic,
                                     weight=self._flood_weight(DRY, hard=False))
            dry_route, _, _ = self._measure(dry_path, DRY)
            detour = route.minutes - dry_route.minutes
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No dry-weather baseline to compare against; report the route, not a
            # detour figure we cannot honestly compute.
            detour = 0.0
        return ResilientRoute(route, level, status, crossed, deepest, detour)

    # -- diagnostics -----------------------------------------------------

    def summary(self) -> str:
        components = nx.number_weakly_connected_components(self.graph)
        return (f"{self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} directed edges, "
                f"{components} weak components, "
                f"routable core {len(self.routable)} nodes "
                f"({100 * len(self.routable) / max(self.graph.number_of_nodes(), 1):.0f}%)")
