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


class RoadNetwork:
    """A directed street graph weighted by travel time."""

    def __init__(self, graph: nx.DiGraph, node_xy: np.ndarray, frame: LocalFrame):
        self.graph = graph
        self.node_xy = node_xy
        self.frame = frame
        self.routable = self._find_routable_core()

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, frame: LocalFrame) -> "RoadNetwork":
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
        return cls(graph, node_xy, frame)

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
            data["time"] = data["base_time"] * factor
            affected += 1
        return affected

    def clear_congestion(self) -> None:
        for _, _, data in self.graph.edges(data=True):
            data["congestion"] = 1.0
            data["time"] = data["base_time"]

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

    # -- diagnostics -----------------------------------------------------

    def summary(self) -> str:
        components = nx.number_weakly_connected_components(self.graph)
        return (f"{self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} directed edges, "
                f"{components} weak components, "
                f"routable core {len(self.routable)} nodes "
                f"({100 * len(self.routable) / max(self.graph.number_of_nodes(), 1):.0f}%)")
