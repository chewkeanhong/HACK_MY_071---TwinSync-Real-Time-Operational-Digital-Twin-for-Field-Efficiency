"""3D line-of-sight coverage -- the reason this twin has to be three-dimensional.

A 2D dispatch map answers "which buildings are within range of the tower". That question
has the wrong answer in a dense CBD, because a 200 m tower sits between the antenna and
half the buildings inside that radius. Those buildings are in range and dark.

Here a building is covered only if a ray from the antenna actually reaches its facade
without passing through another building's volume. The gap between the two answers is the
number the pitch is built on, and :meth:`CoverageEngine.compare_2d_vs_3d` reports it.

Coverage is precomputed once at startup and cached, so a tower failing at demo time is a
set difference rather than a recomputation.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import segment_ring_crossings
from .world import World

# Sample points per building: a few spots around the facade, at a few heights. A
# building counts as covered if enough of them can see the antenna.
FACADE_SAMPLES = 6
HEIGHT_FRACTIONS = (0.05, 0.5, 0.9)
COVERED_THRESHOLD = 1.0 / 3.0

# Receiver height offset so ground-floor samples are not buried in the pavement.
GROUND_CLEARANCE_M = 1.5


@dataclass
class TowerCoverage:
    """Which buildings one tower can actually serve."""

    tower_id: str
    covered: set[str] = field(default_factory=set)
    in_radius: set[str] = field(default_factory=set)   # what a 2D model would claim
    visibility: dict[str, float] = field(default_factory=dict)

    @property
    def blocked(self) -> set[str]:
        """In range on a flat map, but not actually reachable in 3D."""
        return self.in_radius - self.covered


class CoverageEngine:
    """Precomputes and serves tower -> building visibility for the whole scene."""

    def __init__(self, world: World):
        self.world = world
        self.by_tower: dict[str, TowerCoverage] = {}
        self._served_by: dict[str, set[str]] = {}   # building id -> tower ids

    # -- raycasting ------------------------------------------------------

    def _is_blocked(self, origin_xy: np.ndarray, origin_z: float,
                    target_xy: np.ndarray, target_z: float,
                    ignore: set[int]) -> bool:
        """True if any building's volume intersects the 3D segment.

        The ray height varies linearly along the segment, so within a footprint the
        lowest the ray gets is at one of the two crossing points -- comparing the roof
        against those two heights is exact for a flat-roofed extrusion.
        """
        polygons = self.world.polygons
        for index in polygons.candidates_near_segment(origin_xy, target_xy):
            if index in ignore:
                continue
            roof = polygons.heights[index]
            if roof <= min(origin_z, target_z):
                continue    # too short to block this ray anywhere

            crossings = segment_ring_crossings(origin_xy, target_xy, polygons.ring(index))
            if len(crossings) < 1:
                continue

            t_enter, t_exit = crossings[0], crossings[-1]
            z_enter = origin_z + t_enter * (target_z - origin_z)
            z_exit = origin_z + t_exit * (target_z - origin_z)
            if min(z_enter, z_exit) < roof:
                return True
        return False

    def _sample_points(self, building_index: int) -> list[tuple[np.ndarray, float]]:
        """Points on the facade at several heights, as receiver positions."""
        ring = self.world.polygons.ring(building_index)[:-1]   # drop closing vertex
        height = float(self.world.polygons.heights[building_index])

        step = max(1, len(ring) // FACADE_SAMPLES)
        facade = ring[::step][:FACADE_SAMPLES]

        points = []
        for fraction in HEIGHT_FRACTIONS:
            z = max(GROUND_CLEARANCE_M, height * fraction)
            for point in facade:
                points.append((point, z))
        return points

    # -- precomputation --------------------------------------------------

    def compute(self, *, verbose: bool = True) -> "CoverageEngine":
        started = time.time()
        world = self.world
        self.by_tower.clear()
        self._served_by = {b.id: set() for b in world.buildings}

        for tower in world.towers:
            result = TowerCoverage(tower_id=tower.id)
            host_index = (world.building(tower.host_building).index
                          if tower.host_building else None)

            nearby = world.polygons.candidates_within(tower.xy, tower.range_m)
            for index in nearby:
                building = world.buildings[index]
                distance = float(np.hypot(*(building.centroid_xy - tower.xy)))
                if distance > tower.range_m:
                    continue

                # What a flat 2D coverage radius would claim, for comparison.
                result.in_radius.add(building.id)

                ignore = {index} | ({host_index} if host_index is not None else set())
                samples = self._sample_points(index)
                visible = sum(
                    0 if self._is_blocked(tower.xy, tower.antenna_height, point, z, ignore)
                    else 1
                    for point, z in samples
                )
                fraction = visible / len(samples)
                if fraction > 0:
                    result.visibility[building.id] = round(fraction, 3)
                if fraction >= COVERED_THRESHOLD:
                    result.covered.add(building.id)
                    self._served_by[building.id].add(tower.id)

            self.by_tower[tower.id] = result
            if verbose:
                print(f"  {tower.id:<10} in-radius {len(result.in_radius):4d}  "
                      f"3D-covered {len(result.covered):4d}  "
                      f"blocked {len(result.blocked):4d}")

        if verbose:
            print(f"  coverage computed in {time.time() - started:.1f}s")
        return self

    # -- queries ---------------------------------------------------------

    def served_by(self, building_id: str) -> set[str]:
        """Towers that can currently reach this building (ignoring their status)."""
        return self._served_by.get(building_id, set())

    def outage(self, failed_tower_ids: set[str]) -> set[str]:
        """Buildings left with no healthy tower once these towers fail.

        This is the set difference the precomputation exists to make cheap.
        """
        dark = set()
        for tower_id in failed_tower_ids:
            coverage = self.by_tower.get(tower_id)
            if coverage is None:
                continue
            for building_id in coverage.covered:
                if not (self._served_by[building_id] - failed_tower_ids):
                    dark.add(building_id)
        return dark

    def outage_2d(self, failed_tower_ids: set[str]) -> set[str]:
        """What a *fair* 2D model concludes goes dark: radius in, radius out.

        This is the comparison that matters, and it is not the flattering one. Simply
        counting everything inside a failed tower's circle overstates wildly (723
        buildings against a true 40) -- but no competent operator reasons that way; they
        also see the neighbouring cells' circles overlapping the same ground.

        Do it properly and a 2D model concludes that **nobody** is affected, because
        every building inside the failed circle also sits inside some healthy one. It is
        wrong, and it is wrong in the dangerous direction: those buildings cannot see the
        neighbouring tower, because a 200 m tower is in the way. The flat map does not
        overstate the outage. It misses it entirely.
        """
        healthy = {t.id for t in self.world.towers} - failed_tower_ids
        dark = set()
        for tower_id in failed_tower_ids:
            entry = self.by_tower.get(tower_id)
            if entry is None:
                continue
            for building_id in entry.in_radius:
                if not any(building_id in self.by_tower[h].in_radius for h in healthy):
                    dark.add(building_id)
        return dark

    def naive_radius(self, failed_tower_ids: set[str]) -> set[str]:
        """Everything inside the failed towers' circles, ignoring neighbours entirely."""
        naive = set()
        for tower_id in failed_tower_ids:
            entry = self.by_tower.get(tower_id)
            if entry:
                naive |= entry.in_radius
        return naive

    def subscribers_affected(self, building_ids: set[str]) -> int:
        return sum(self.world.building(b).subscribers for b in building_ids)

    def compare_2d_vs_3d(self, tower_id: str) -> dict:
        """The headline demo statistic for a single tower."""
        coverage = self.by_tower[tower_id]
        naive = coverage.in_radius
        real = coverage.covered
        return {
            "tower": tower_id,
            "buildings_2d": len(naive),
            "buildings_3d": len(real),
            "blocked": len(coverage.blocked),
            "subscribers_2d": self.subscribers_affected(naive),
            "subscribers_3d": self.subscribers_affected(real),
        }

    # -- caching ---------------------------------------------------------

    def _fingerprint(self) -> str:
        """Identifies the scene this cache was computed against.

        Tower ids alone are not enough: re-running the height imputation changes the
        building extrusions, which changes every ray, while leaving the tower list
        identical. A cache keyed only on tower ids would be silently reused and the
        outage shadow would no longer match the city on screen.
        """
        heights = self.world.polygons.heights
        digest = hashlib.sha256()
        digest.update(",".join(sorted(t.id for t in self.world.towers)).encode())
        digest.update(f"|{len(self.world.buildings)}|".encode())
        digest.update(np.round(heights, 2).tobytes())
        for tower in sorted(self.world.towers, key=lambda t: t.id):
            digest.update(f"|{tower.antenna_height:.2f}:{tower.range_m:.1f}".encode())
        return digest.hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fingerprint": self._fingerprint(),
            "towers": {
                tower_id: {
                    "covered": sorted(c.covered),
                    "in_radius": sorted(c.in_radius),
                    "visibility": c.visibility,
                }
                for tower_id, c in self.by_tower.items()
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    def load(self, path: str | Path) -> bool:
        """Restore a cached computation. Returns False if it does not match the scene."""
        path = Path(path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != self._fingerprint():
            return False
        towers = payload.get("towers", {})
        if set(towers) != {t.id for t in self.world.towers}:
            return False

        self.by_tower = {}
        self._served_by = {b.id: set() for b in self.world.buildings}
        for tower_id, entry in towers.items():
            coverage = TowerCoverage(
                tower_id=tower_id,
                covered=set(entry["covered"]),
                in_radius=set(entry["in_radius"]),
                visibility=entry.get("visibility", {}),
            )
            self.by_tower[tower_id] = coverage
            for building_id in coverage.covered:
                if building_id in self._served_by:
                    self._served_by[building_id].add(tower_id)
        return True
