"""Geometry primitives for the TwinSync twin.

Deliberately built on plain numpy rather than geopandas/GDAL: the whole AOI is a couple
of kilometres across, so a local equirectangular projection is accurate to well under a
metre and lets every hot path run as flat array maths.

The two things everything else is built on:

* :class:`LocalFrame` -- lon/lat <-> local metres, so distances are just Pythagoras.
* :class:`PolygonSet` -- all building footprints in one flat vertex array, with a
  uniform grid index, so ray-vs-city queries touch only nearby buildings.
"""

from __future__ import annotations

import math

import numpy as np

EARTH_RADIUS_M = 6_371_000.0

# Derived from the same sphere haversine_m uses, deliberately. Mixing ellipsoidal
# constants here with a spherical haversine puts a ~0.5% disagreement between two
# functions that are supposed to measure the same thing. Absolute scale is off by that
# much versus WGS84, but every distance in the twin -- ranges, ETAs, routing costs -- is
# compared against another distance from the same frame, so consistency is what counts.
METRES_PER_DEG = EARTH_RADIUS_M * math.pi / 180.0   # 111_195 m
METRES_PER_DEG_LAT = METRES_PER_DEG
METRES_PER_DEG_LON_EQ = METRES_PER_DEG


# ---------------------------------------------------------------- projection


class LocalFrame:
    """Equirectangular lon/lat <-> metres about a fixed origin.

    Accurate to a few centimetres over a city-block-scale AOI, which is far below
    the uncertainty in the building heights we are modelling.
    """

    def __init__(self, origin_lon: float, origin_lat: float):
        self.origin_lon = float(origin_lon)
        self.origin_lat = float(origin_lat)
        self._mx = METRES_PER_DEG_LON_EQ * math.cos(math.radians(self.origin_lat))
        self._my = METRES_PER_DEG_LAT

    @classmethod
    def from_bounds(cls, min_lon, min_lat, max_lon, max_lat) -> "LocalFrame":
        return cls((min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0)

    def to_xy(self, lon, lat):
        """lon/lat (scalars or arrays) -> x/y metres east/north of the origin."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        return (lon - self.origin_lon) * self._mx, (lat - self.origin_lat) * self._my

    def to_lonlat(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return x / self._mx + self.origin_lon, y / self._my + self.origin_lat


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres. Used for sanity checks and OSM-space work."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ------------------------------------------------------------- ring geometry


def ring_area(ring: np.ndarray) -> float:
    """Shoelace area of a closed ring of (x, y) metres."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def ring_perimeter(ring: np.ndarray) -> float:
    return float(np.hypot(*np.diff(ring, axis=0).T).sum())


def ring_centroid(ring: np.ndarray) -> tuple[float, float]:
    """Area-weighted centroid; falls back to the vertex mean for degenerate rings."""
    x, y = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    cross = x * y1 - x1 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-9:
        return float(ring[:, 0].mean()), float(ring[:, 1].mean())
    return (float(((x + x1) * cross).sum() / (6.0 * area)),
            float(((y + y1) * cross).sum() / (6.0 * area)))


def point_in_ring(px: float, py: float, ring: np.ndarray) -> bool:
    """Even-odd ray casting test for a single point against a closed ring."""
    x0, y0 = ring[:-1, 0], ring[:-1, 1]
    x1, y1 = ring[1:, 0], ring[1:, 1]
    straddles = (y0 > py) != (y1 > py)
    if not straddles.any():
        return False
    # x of each straddling edge at height py
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
    return bool(np.count_nonzero(straddles & (px < x_cross)) % 2 == 1)


def segment_ring_crossings(p0: np.ndarray, p1: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Parameters t in [0, 1] where segment p0->p1 crosses the ring's edges.

    Standard 2D segment-segment intersection, vectorised over all edges at once.
    """
    d = p1 - p0
    e0 = ring[:-1]
    e1 = ring[1:]
    e = e1 - e0

    denom = d[0] * e[:, 1] - d[1] * e[:, 0]
    parallel = np.abs(denom) < 1e-12
    safe = np.where(parallel, 1.0, denom)

    w = e0 - p0
    t = (w[:, 0] * e[:, 1] - w[:, 1] * e[:, 0]) / safe   # along the ray
    u = (w[:, 0] * d[1] - w[:, 1] * d[0]) / safe         # along the edge

    hit = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return np.sort(t[hit])


# ------------------------------------------------------------- polygon set


class PolygonSet:
    """Every building footprint packed into one flat array, plus a uniform grid index.

    Storing rings contiguously (``verts`` + ``offsets``) instead of as a list of arrays
    keeps the ray-casting inner loop cheap, and the grid means a single ray only tests
    the handful of buildings it actually passes near.
    """

    def __init__(self, rings: list[np.ndarray], heights: np.ndarray, cell_size: float = 100.0):
        self.count = len(rings)
        self.offsets = np.zeros(self.count + 1, dtype=np.int64)
        for i, ring in enumerate(rings):
            self.offsets[i + 1] = self.offsets[i] + len(ring)
        self.verts = (np.concatenate(rings, axis=0) if rings
                      else np.zeros((0, 2), dtype=np.float64))
        self.heights = np.asarray(heights, dtype=np.float64)

        self.bboxes = np.zeros((self.count, 4), dtype=np.float64)
        for i in range(self.count):
            ring = self.ring(i)
            self.bboxes[i] = (ring[:, 0].min(), ring[:, 1].min(),
                              ring[:, 0].max(), ring[:, 1].max())

        self.cell_size = float(cell_size)
        self._build_grid()

    def ring(self, i: int) -> np.ndarray:
        return self.verts[self.offsets[i]:self.offsets[i + 1]]

    # -- spatial index ---------------------------------------------------

    def _build_grid(self) -> None:
        if self.count == 0:
            self.grid_origin = (0.0, 0.0)
            self.grid_dims = (0, 0)
            self._cells = {}
            return

        self.grid_origin = (float(self.bboxes[:, 0].min()), float(self.bboxes[:, 1].min()))
        max_x, max_y = float(self.bboxes[:, 2].max()), float(self.bboxes[:, 3].max())
        self.grid_dims = (
            int((max_x - self.grid_origin[0]) // self.cell_size) + 1,
            int((max_y - self.grid_origin[1]) // self.cell_size) + 1,
        )

        cells: dict[tuple[int, int], list[int]] = {}
        for i in range(self.count):
            min_x, min_y, mx, my = self.bboxes[i]
            for cx in range(self._cell_x(min_x), self._cell_x(mx) + 1):
                for cy in range(self._cell_y(min_y), self._cell_y(my) + 1):
                    cells.setdefault((cx, cy), []).append(i)
        self._cells = {k: np.array(v, dtype=np.int64) for k, v in cells.items()}

    def _cell_x(self, x: float) -> int:
        return int((x - self.grid_origin[0]) // self.cell_size)

    def _cell_y(self, y: float) -> int:
        return int((y - self.grid_origin[1]) // self.cell_size)

    def candidates_near_segment(self, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
        """Building indices whose bbox could possibly meet the segment p0->p1."""
        if self.count == 0:
            return np.zeros(0, dtype=np.int64)

        lo_x, hi_x = sorted((p0[0], p1[0]))
        lo_y, hi_y = sorted((p0[1], p1[1]))

        found: list[np.ndarray] = []
        for cx in range(self._cell_x(lo_x), self._cell_x(hi_x) + 1):
            for cy in range(self._cell_y(lo_y), self._cell_y(hi_y) + 1):
                bucket = self._cells.get((cx, cy))
                if bucket is not None:
                    found.append(bucket)
        if not found:
            return np.zeros(0, dtype=np.int64)

        ids = np.unique(np.concatenate(found))
        box = self.bboxes[ids]
        keep = ((box[:, 0] <= hi_x) & (box[:, 2] >= lo_x)
                & (box[:, 1] <= hi_y) & (box[:, 3] >= lo_y))
        return ids[keep]

    def candidates_within(self, centre: np.ndarray, radius: float) -> np.ndarray:
        """Building indices whose bbox overlaps the square around ``centre``."""
        if self.count == 0:
            return np.zeros(0, dtype=np.int64)
        lo_x, hi_x = centre[0] - radius, centre[0] + radius
        lo_y, hi_y = centre[1] - radius, centre[1] + radius

        found: list[np.ndarray] = []
        for cx in range(self._cell_x(lo_x), self._cell_x(hi_x) + 1):
            for cy in range(self._cell_y(lo_y), self._cell_y(hi_y) + 1):
                bucket = self._cells.get((cx, cy))
                if bucket is not None:
                    found.append(bucket)
        if not found:
            return np.zeros(0, dtype=np.int64)

        ids = np.unique(np.concatenate(found))
        box = self.bboxes[ids]
        keep = ((box[:, 0] <= hi_x) & (box[:, 2] >= lo_x)
                & (box[:, 1] <= hi_y) & (box[:, 3] >= lo_y))
        return ids[keep]
