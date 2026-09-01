"""ST-DBSCAN fault localisation.

When a cascade takes out several sites, an operator does not want N tickets. They want
one: *this area, starting here, spreading this way*. That is a clustering problem in
space and time at once, and it is the problem ST-DBSCAN (Birant & Kut, 2007) exists to
solve -- ordinary DBSCAN cannot express "close together **and** close in time", because
it has only one radius to spend.

The implementation here is the real algorithm, not a similarity heuristic:

* Two radii. ``eps_spatial`` in metres and ``eps_temporal`` in seconds, tested
  independently. An alarm 200 m away but forty minutes later is not a neighbour.
* Genuine core / border / noise semantics. An alarm with too few neighbours is labelled
  ``-1`` and reported as isolated. **This is the behaviour that matters for the demo**:
  a stub that groups every failed tower can never say "these two are one incident and
  that third one is unrelated", which is the entire operational value.
* A non-spatial attribute gate. Birant & Kut use a threshold on the non-spatial value to
  stop clusters of dissimilar character merging; the equivalent here is a fault-family
  compatibility table, because "amplifier overheating" and "backhaul saturated on
  another site" are not one event no matter how close they happen to be.

Distance is **3D**. The Z coordinate is the antenna phase centre -- DEM ground elevation
plus mast height -- so two sites at the same map position but 200 m apart vertically are
correctly far apart. That is what makes this GeoAI rather than a scatter plot with a
clock attached.

The streaming wrapper :class:`AlarmClusterer` is what the simulation actually calls; see
its docstring for why re-clustering (rather than incremental insertion) is the right
call at fleet scale, and how cluster ids are kept stable across re-runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Two alarms this far apart in space are never neighbours, regardless of timing.
#
# 900 m is tuned against the KL fleet: the closest pair of distinct sites is ~290 m and
# the median nearest-neighbour spacing is ~400 m, so this reaches roughly two sites out
# -- wide enough to catch a genuine cascade along a feeder, tight enough that half the
# CBD does not land in one cluster. Raise it and everything merges; drop it below ~350 m
# and even adjacent sites stop grouping.
DEFAULT_EPS_SPATIAL_M = 900.0

# ...and this far apart in time. Ten minutes is the window in which a shared root cause
# is still a plausible explanation. Beyond it, two failures in the same street are
# better modelled as two incidents that happen to be neighbours.
DEFAULT_EPS_TEMPORAL_S = 600.0

# Minimum alarms (including the point itself) for a core point. Two is deliberate: this
# is a 15-site fleet, and requiring three would mean a genuine two-site cascade is
# reported as two unrelated tickets -- the exact failure we are trying to remove.
DEFAULT_MIN_PTS = 2

# How long an alarm stays eligible for clustering. Past this it is history: it stops
# pulling new alarms into its cluster, which stops one long incident from slowly
# accreting every unrelated fault of the afternoon.
DEFAULT_RETENTION_S = 1800.0

# Fraction of members two cluster snapshots must share to be considered "the same
# cluster" across a re-run, so its id survives. See AlarmClusterer._stable_id.
JACCARD_REUSE = 0.5

NOISE = -1

# Which fault profiles can plausibly belong to one physical cascade, grouped by the
# shared root cause an operator would actually go looking for.
#
# * rf_path   -- amplifier_degradation, antenna_misalignment. Both sit on the radiating
#   chain. A squall that torques an antenna also drives heat and humidity into the
#   amplifier behind it, so under one weather cell these co-occur.
# * power     -- power_failure. Mains loss.
# * transport -- backhaul_congestion. Transmission saturated.
FAULT_FAMILY = {
    "amplifier_degradation": "rf_path",
    "antenna_misalignment": "rf_path",
    "power_failure": "power",
    "backhaul_congestion": "transport",
}
DEFAULT_FAMILY = "rf_path"

# Symmetric compatibility between families. DBSCAN's core/border reasoning assumes the
# neighbour relation is symmetric, so this table must be too -- an asymmetric "A can
# cause B" relation would make cluster membership depend on visit order.
#
# power <-> transport: a site losing mains dumps its traffic onto its neighbours, and
#   the first thing to saturate is their backhaul. The textbook telecom cascade.
# power <-> rf_path:   mains disturbance damages amplifiers, so a power event is a
#   plausible head for an RF cascade.
# rf_path <-> transport is deliberately absent. A bent antenna here and a congested
#   backhaul there share no root cause worth putting on one ticket.
COMPATIBLE_FAMILIES = {
    frozenset({"rf_path"}),
    frozenset({"power"}),
    frozenset({"transport"}),
    frozenset({"power", "transport"}),
    frozenset({"power", "rf_path"}),
}


def families_compatible(a: str, b: str) -> bool:
    fa = FAULT_FAMILY.get(a, DEFAULT_FAMILY)
    fb = FAULT_FAMILY.get(b, DEFAULT_FAMILY)
    return frozenset({fa, fb}) in COMPATIBLE_FAMILIES


@dataclass(frozen=True)
class Alarm:
    """One tower's state change, as a point in space-time.

    ``z`` is the antenna phase centre above sea level, not the mast height.
    """

    tower_id: str
    x: float
    y: float
    z: float
    t: float
    alarm_type: str

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class LocalisationResult:
    """What the localiser concluded about one alarm."""

    cluster_id: str
    members: list[str] = field(default_factory=list)
    is_noise: bool = False
    span_m: float = 0.0
    span_s: float = 0.0
    model: str = "st-dbscan-v1"

    def describe(self) -> str:
        if self.is_noise:
            return "isolated (no co-located cascade)"
        return (f"{len(self.members)} sites within {self.span_m:.0f} m "
                f"over {self.span_s:.0f} s")


def st_dbscan(alarms: list[Alarm], *,
              eps_spatial: float = DEFAULT_EPS_SPATIAL_M,
              eps_temporal: float = DEFAULT_EPS_TEMPORAL_S,
              min_pts: int = DEFAULT_MIN_PTS) -> list[int]:
    """Cluster alarms in space and time. Returns one label per alarm, ``-1`` for noise.

    Standard DBSCAN expansion over a neighbour relation that requires spatial, temporal
    and fault-family agreement simultaneously. The fleet is small enough that building
    the full neighbour matrix is cheaper than any index, and it keeps the predicate in
    one readable place.
    """
    n = len(alarms)
    if n == 0:
        return []

    points = np.array([[a.x, a.y, a.z] for a in alarms], dtype=np.float64)
    times = np.array([a.t for a in alarms], dtype=np.float64)

    # 3D euclidean distance, every pair at once.
    delta = points[:, None, :] - points[None, :, :]
    spatial = np.sqrt((delta * delta).sum(axis=-1))
    temporal = np.abs(times[:, None] - times[None, :])

    neighbours = (spatial <= eps_spatial) & (temporal <= eps_temporal)
    for i in range(n):
        for j in range(i + 1, n):
            if neighbours[i, j] and not families_compatible(
                    alarms[i].alarm_type, alarms[j].alarm_type):
                neighbours[i, j] = neighbours[j, i] = False

    labels = [NOISE] * n
    visited = [False] * n
    cluster = 0

    for start in range(n):
        if visited[start]:
            continue
        visited[start] = True

        seeds = list(np.flatnonzero(neighbours[start]))
        if len(seeds) < min_pts:
            continue        # not a core point; may still be claimed as a border point

        labels[start] = cluster
        queue = [s for s in seeds if s != start]
        while queue:
            current = queue.pop(0)
            if not visited[current]:
                visited[current] = True
                current_neighbours = list(np.flatnonzero(neighbours[current]))
                # Only a core point extends the cluster. A border point joins it but
                # does not drag its own neighbourhood in -- that distinction is what
                # stops two clusters bridging through a single sparse alarm.
                if len(current_neighbours) >= min_pts:
                    queue.extend(c for c in current_neighbours if labels[c] == NOISE)
            if labels[current] == NOISE:
                labels[current] = cluster
        cluster += 1

    return labels


class AlarmClusterer:
    """Streaming ST-DBSCAN over a rolling window of recent alarms.

    Re-clusters from scratch on every new alarm rather than inserting incrementally.
    That sounds wasteful and is not: the retention window holds at most a handful of
    alarms for a 15-site fleet, the whole pass is well under a millisecond, and it
    avoids the class of bug where an incrementally-maintained cluster diverges from what
    a fresh run would produce.

    The cost of re-clustering is that raw labels are just indices and can renumber
    between runs, which would make cluster ids flicker on screen. :meth:`_stable_id`
    fixes that by matching each new cluster to the previous snapshot on membership
    overlap and reusing the old public id when they are substantially the same set.
    """

    def __init__(self, world, *,
                 eps_spatial: float = DEFAULT_EPS_SPATIAL_M,
                 eps_temporal: float = DEFAULT_EPS_TEMPORAL_S,
                 min_pts: int = DEFAULT_MIN_PTS,
                 retention_s: float = DEFAULT_RETENTION_S):
        self.world = world
        self.eps_spatial = float(eps_spatial)
        self.eps_temporal = float(eps_temporal)
        self.min_pts = int(min_pts)
        self.retention_s = float(retention_s)

        self._alarms: list[Alarm] = []
        self._previous: dict[str, set[str]] = {}    # public id -> member tower ids
        self._counter = 0

    # -- ingest ----------------------------------------------------------

    def observe(self, tower_id: str, alarm_type: str, t: float) -> LocalisationResult:
        """Record an alarm and return the localisation verdict for it."""
        tower = self.world.tower(tower_id)
        self._alarms = [a for a in self._alarms
                        if t - a.t <= self.retention_s and a.tower_id != tower_id]
        self._alarms.append(Alarm(
            tower_id=tower_id,
            x=float(tower.xy[0]), y=float(tower.xy[1]),
            # Antenna phase centre above sea level: this is where the DEM enters the
            # clustering, and why two masts of equal height on different ground are not
            # at the same point.
            z=float(tower.antenna_z),
            t=float(t),
            alarm_type=alarm_type,
        ))
        return self._localise(tower_id)

    def forget(self, tower_id: str) -> None:
        """Drop a tower's alarm once its incident is resolved."""
        self._alarms = [a for a in self._alarms if a.tower_id != tower_id]

    # -- clustering ------------------------------------------------------

    def _localise(self, tower_id: str) -> LocalisationResult:
        labels = st_dbscan(self._alarms,
                           eps_spatial=self.eps_spatial,
                           eps_temporal=self.eps_temporal,
                           min_pts=self.min_pts)

        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            if label != NOISE:
                groups.setdefault(label, []).append(index)

        assigned = self._stable_id({
            label: {self._alarms[i].tower_id for i in members}
            for label, members in groups.items()
        })
        self._previous = {
            public: {self._alarms[i].tower_id for i in groups[label]}
            for label, public in assigned.items()
        }

        position = next(i for i, a in enumerate(self._alarms) if a.tower_id == tower_id)
        label = labels[position]
        if label == NOISE:
            return LocalisationResult(cluster_id="ISOLATED", members=[tower_id],
                                      is_noise=True)

        indices = groups[label]
        members = sorted(self._alarms[i].tower_id for i in indices)
        return LocalisationResult(
            cluster_id=assigned[label],
            members=members,
            is_noise=False,
            span_m=self._span_m(indices),
            span_s=self._span_s(indices),
        )

    def _span_m(self, indices: list[int]) -> float:
        """Widest 3D separation inside the cluster -- how big the affected zone is."""
        if len(indices) < 2:
            return 0.0
        points = np.array([self._alarms[i].xyz for i in indices])
        delta = points[:, None, :] - points[None, :, :]
        return float(np.sqrt((delta * delta).sum(axis=-1)).max())

    def _span_s(self, indices: list[int]) -> float:
        times = [self._alarms[i].t for i in indices]
        return float(max(times) - min(times))

    def _stable_id(self, groups: dict[int, set[str]]) -> dict[int, str]:
        """Map this run's raw labels onto public ids, reusing them where possible.

        Without this an incident's cluster id changes every time an unrelated alarm
        arrives and shifts the numbering, and the operator watching the screen sees the
        identifier of the thing they are working on mutate underneath them.
        """
        assigned: dict[int, str] = {}
        claimed: set[str] = set()

        for label, members in sorted(groups.items()):
            best_id, best_score = None, 0.0
            for public, previous in self._previous.items():
                if public in claimed:
                    continue
                union = members | previous
                if not union:
                    continue
                score = len(members & previous) / len(union)
                if score > best_score:
                    best_id, best_score = public, score
            if best_id is not None and best_score >= JACCARD_REUSE:
                assigned[label] = best_id
                claimed.add(best_id)
            else:
                self._counter += 1
                assigned[label] = f"CL-{self._counter:03d}"
        return assigned
