"""Crew assignment, batching and dynamic rerouting.

Three behaviours here are the efficiency story:

1. **Assign on travel time.** The nearest crew in metres is often not the first to arrive.
2. **Batch nearby work.** A second fault close to one already being driven to is folded
   into the same trip. Each fold is one truck roll that never happens.
3. **Reroute when priority changes.** A crew driving to a car park gets turned around when
   a hospital goes dark, instead of finishing the low-value job first.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from .priority import Impact, priority_score
from .routing import RoadNetwork, Route

# A second fault is worth folding into an existing trip if the detour is small.
BATCH_DETOUR_LIMIT_S = 300.0

# Reassigning a crew mid-drive is disruptive; only do it for a clearly bigger job.
PREEMPT_MARGIN = 1.35

# How long a crew spends on site, by severity.
REPAIR_SECONDS = {"degraded": 20 * 60.0, "down": 35 * 60.0}
DEFAULT_REPAIR_SECONDS = 30 * 60.0


@dataclass
class Incident:
    """A fault the twin knows about."""

    id: str
    tower_id: str
    severity: str
    detected_at: float                 # when the operator learned about it
    impact: Impact
    xy: np.ndarray
    # When service actually degraded. Subscribers are already suffering between this and
    # detected_at, so every customer-facing metric has to run from here.
    fault_started_at: float | None = None
    priority: float = 0.0
    assigned_to: str | None = None
    resolved_at: float | None = None
    repair_started_at: float | None = None

    # Which crew a flat map would have picked (nearest in metres) and what that choice
    # would actually have cost in driving time. Recorded so the 2D view can show its
    # own answer rather than borrowing TwinSync's.
    crew_2d: str | None = None
    crew_2d_minutes: float | None = None
    assigned_minutes: float | None = None
    ai_cluster_id: str | None = None
    ai_cluster_members: list[str] = field(default_factory=list)
    # True when ST-DBSCAN found no co-located, contemporaneous, compatible cascade --
    # i.e. this really is a standalone fault rather than part of a group.
    ai_cluster_noise: bool = False
    ai_cluster_span_m: float = 0.0
    ai_cluster_span_s: float = 0.0
    # When the cluster this incident belongs to was first localised. Time-to-localise is
    # measured from the fault starting to this instant.
    ai_localised_at: float | None = None
    ai_risk_score: float = 0.0
    ai_risk_band: str = "low"
    ai_risk_factors: list[dict] = field(default_factory=list)
    # "none" until a model has actually scored this incident. The baseline arm runs no
    # localiser and no risk model at all, and its incidents must not carry a model tag
    # implying otherwise -- that would quietly credit the A/B's control group with the
    # thing being tested.
    ai_model_source: str = "none"

    @property
    def resolved(self) -> bool:
        return self.resolved_at is not None

    def refresh_priority(self, now: float) -> float:
        self.priority = priority_score(
            self.impact, self.severity, (now - self.detected_at) / 60.0
        )
        return self.priority


@dataclass
class Crew:
    """A field team and the trip it is currently on."""

    id: str
    name: str
    home_xy: np.ndarray
    xy: np.ndarray
    status: str = "idle"               # idle | en_route | on_site
    queue: list[str] = field(default_factory=list)   # incident ids, in visit order
    route: Route | None = None
    route_progress_s: float = 0.0
    eta_s: float = 0.0
    on_site_until: float | None = None
    trips: int = 0                     # completed truck rolls
    jobs_done: int = 0
    # Odometer and clock, accumulated as legs are dispatched and worked. These are what
    # the fuel, CO2 and crew-utilisation figures are computed from, so they have to come
    # from the routes actually driven rather than from straight-line estimates.
    distance_m: float = 0.0
    driving_s: float = 0.0
    on_site_s: float = 0.0

    @property
    def current_job(self) -> str | None:
        return self.queue[0] if self.queue else None

    @property
    def busy(self) -> bool:
        return bool(self.queue)


class DispatchEngine:
    """Owns crews and incidents, and decides who goes where.

    ``smart=False`` reproduces how this is done today: work the queue in the order it
    arrived, send whoever is nearest as the crow flies, never combine trips, never turn a
    crew around. The demo runs the identical scenario through both modes so the
    improvement figures are measured rather than asserted.
    """

    def __init__(self, network: RoadNetwork, crews: list[Crew], *, smart: bool = True):
        self.network = network
        self.crews = crews
        self.smart = smart
        self.incidents: dict[str, Incident] = {}
        self.log: list[tuple[float, str]] = []
        self._batched = 0
        self._reassignments = 0
        self._counter = itertools.count(1)
        # Unassignable incidents are retried on a timer; without this they would repeat
        # the same "no crew available" line every retry and bury the real timeline.
        self._deferred: set[str] = set()

    # -- bookkeeping -----------------------------------------------------

    def _note(self, now: float, message: str) -> None:
        self.log.append((now, message))

    def _note_once(self, now: float, incident_id: str, message: str) -> None:
        """Log a deferral only the first time, not on every retry."""
        if incident_id in self._deferred:
            return
        self._deferred.add(incident_id)
        self._note(now, message)

    def crew(self, crew_id: str) -> Crew:
        return next(c for c in self.crews if c.id == crew_id)

    @property
    def open_incidents(self) -> list[Incident]:
        return [i for i in self.incidents.values() if not i.resolved]

    @property
    def batched_count(self) -> int:
        """Jobs folded into an existing trip -- i.e. truck rolls avoided."""
        return self._batched

    @property
    def reassignment_count(self) -> int:
        return self._reassignments

    # -- incident intake -------------------------------------------------

    def report(self, now: float, tower_id: str, severity: str, impact: Impact,
               xy: np.ndarray, fault_started_at: float | None = None) -> Incident:
        incident = Incident(
            id=f"INC-{next(self._counter):03d}",
            tower_id=tower_id,
            severity=severity,
            detected_at=now,
            impact=impact,
            xy=np.asarray(xy, dtype=np.float64),
            fault_started_at=fault_started_at if fault_started_at is not None else now,
        )
        incident.refresh_priority(now)
        self.incidents[incident.id] = incident
        self._note(now, f"{incident.id} raised at {tower_id} ({severity}) "
                        f"priority {incident.priority:.1f}")
        return incident

    # -- assignment ------------------------------------------------------

    def _travel_time(self, origin: np.ndarray, destination: np.ndarray) -> float:
        return self.network.travel_time(origin, destination)

    def _batch_candidate(self, incident: Incident) -> tuple[Crew, float] | None:
        """Find a crew already driving somewhere near enough to add this stop.

        The test is the *extra* driving the detour costs, not raw proximity -- two sites
        400 m apart across a river are not neighbours.
        """
        if not self.smart:
            return None     # today's dispatch does one job per trip

        best: tuple[Crew, float] | None = None
        for crew in self.crews:
            if not crew.busy or crew.status == "on_site":
                continue
            current = self.incidents.get(crew.current_job)
            if current is None or current.resolved:
                continue

            detour = self._travel_time(current.xy, incident.xy)
            if detour <= BATCH_DETOUR_LIMIT_S and (best is None or detour < best[1]):
                best = (crew, detour)
        return best

    def assign(self, now: float, incident: Incident) -> str | None:
        """Route the best crew to this incident. Returns the crew id, or None."""
        # 1. Can an in-flight trip absorb it?
        batched = self._batch_candidate(incident)
        if batched is not None:
            crew, detour = batched
            crew.queue.append(incident.id)
            incident.assigned_to = crew.id
            self._batched += 1
            self._note(now, f"{incident.id} batched onto {crew.id}'s current trip "
                            f"(+{detour / 60:.1f} min detour) -- truck roll saved")
            return crew.id

        # 2. Otherwise pick a crew. TwinSync ranks by driving time over the street
        #    graph; today's practice ranks by distance on a flat map.
        idle = [c for c in self.crews if not c.busy]
        pool = idle or self.crews
        if self.smart:
            timings = [(c, self._travel_time(c.xy, incident.xy)) for c in pool]
        else:
            timings = [(c, float(np.hypot(*(c.xy - incident.xy)))) for c in pool]
        timings = [(c, t) for c, t in timings if np.isfinite(t)]
        if not timings:
            self._note_once(now, incident.id, f"{incident.id} unreachable by any crew")
            return None
        timings.sort(key=lambda pair: pair[1])
        chosen, seconds = timings[0]
        if not self.smart:
            # The metric that matters is still time, even when the choice ignored it.
            seconds = self._travel_time(chosen.xy, incident.xy)
            if not np.isfinite(seconds):
                self._note_once(now, incident.id,
                                f"{incident.id} unreachable by any crew")
                return None

        # 3. If every crew is busy, only interrupt one for a materially bigger job.
        if not idle:
            current = self.incidents.get(chosen.current_job)
            if not self.smart:
                self._note_once(now, incident.id,
                                f"{incident.id} queued -- all crews busy")
                return None
            if current and incident.priority < current.priority * PREEMPT_MARGIN:
                self._note_once(now, incident.id,
                                f"{incident.id} queued -- all crews busy on "
                                f"higher-priority work")
                return None
            if current:
                current.assigned_to = None
                chosen.queue.remove(current.id)
                self._reassignments += 1
                self._note(now, f"{chosen.id} pulled off {current.id} "
                                f"(priority {current.priority:.1f}) for {incident.id} "
                                f"(priority {incident.priority:.1f})")

        # Report the honest straight-line comparison, since that is the alternative
        # a dispatcher without this tool would have used.
        nearest_by_distance = min(
            pool, key=lambda c: float(np.hypot(*(c.xy - incident.xy)))
        )
        straight_pick_seconds = self._travel_time(nearest_by_distance.xy, incident.xy)
        incident.crew_2d = nearest_by_distance.id
        incident.crew_2d_minutes = (straight_pick_seconds / 60.0
                                    if np.isfinite(straight_pick_seconds) else None)
        incident.assigned_minutes = seconds / 60.0 if np.isfinite(seconds) else None

        if self.smart and nearest_by_distance.id != chosen.id:
            self._note(now, f"{chosen.id} chosen over {nearest_by_distance.id}: "
                            f"further away, {seconds / 60:.1f} min vs "
                            f"{straight_pick_seconds / 60:.1f} min")

        chosen.queue.insert(0, incident.id)
        incident.assigned_to = chosen.id
        self._deferred.discard(incident.id)
        self._start_leg(now, chosen)
        return chosen.id

    def _start_leg(self, now: float, crew: Crew, *, new_trip: bool = True) -> None:
        """Send a crew driving to the front of its queue.

        ``new_trip`` distinguishes rolling a van out of the depot from continuing to the
        next stop of a trip already under way. Counting the second leg of a batched trip
        as another truck roll would cancel out the very saving batching exists to make.
        """
        target = self.incidents.get(crew.current_job)
        if target is None:
            crew.status = "idle"
            crew.route = None
            return

        route = self.network.route(crew.xy, target.xy)
        crew.route = route
        crew.route_progress_s = 0.0
        crew.eta_s = route.travel_time_s if route else 0.0
        crew.status = "en_route"
        if route is not None:
            # Road distance, not crow-flies. The whole fuel/CO2 argument depends on
            # counting the kilometres actually driven.
            crew.distance_m += route.distance_m
        if new_trip:
            crew.trips += 1
        self._note(now, f"{crew.id} -> {target.id} at {target.tower_id}, "
                        f"ETA {crew.eta_s / 60:.1f} min")

    # -- simulation ------------------------------------------------------

    def tick(self, now: float, dt: float) -> None:
        """Advance crews along their routes and complete repairs."""
        for incident in self.open_incidents:
            incident.refresh_priority(now)

        for crew in self.crews:
            if crew.status == "en_route":
                crew.route_progress_s += dt
                crew.driving_s += dt
                crew.eta_s = max(0.0, (crew.route.travel_time_s if crew.route else 0.0)
                                 - crew.route_progress_s)
                if crew.eta_s <= 0.0:
                    self._arrive(now, crew)

            elif crew.status == "on_site":
                crew.on_site_s += dt
                if crew.on_site_until is not None and now >= crew.on_site_until:
                    self._finish(now, crew)

    def _arrive(self, now: float, crew: Crew) -> None:
        incident = self.incidents.get(crew.current_job)
        if incident is None:
            crew.status = "idle"
            return
        crew.xy = incident.xy.copy()
        crew.status = "on_site"
        crew.route = None
        incident.repair_started_at = now
        crew.on_site_until = now + REPAIR_SECONDS.get(incident.severity,
                                                      DEFAULT_REPAIR_SECONDS)
        self._note(now, f"{crew.id} on site at {incident.tower_id}")

    def _finish(self, now: float, crew: Crew) -> None:
        incident = self.incidents.get(crew.current_job)
        if incident is not None:
            incident.resolved_at = now
            crew.jobs_done += 1
            minutes = (now - incident.detected_at) / 60.0
            self._note(now, f"{incident.id} resolved at {incident.tower_id} "
                            f"after {minutes:.1f} min")
        if crew.queue:
            crew.queue.pop(0)
        crew.on_site_until = None

        if crew.queue:
            # Same van, still out: continuing a batched trip, not a new roll.
            self._start_leg(now, crew, new_trip=False)
        else:
            crew.status = "idle"
            crew.route = None

    def position_of(self, crew: Crew) -> np.ndarray:
        """Where a crew is right now, interpolated along its route for the map."""
        if crew.status != "en_route" or crew.route is None or len(crew.route.xy) < 2:
            return crew.xy

        total = crew.route.travel_time_s
        if total <= 0:
            return crew.xy
        fraction = float(np.clip(crew.route_progress_s / total, 0.0, 1.0))

        # Walk the polyline by cumulative length to find the point at that fraction.
        points = crew.route.xy
        segments = np.hypot(*np.diff(points, axis=0).T)
        cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        distance = fraction * cumulative[-1]
        index = int(np.searchsorted(cumulative, distance, side="right") - 1)
        index = max(0, min(index, len(segments) - 1))

        span = segments[index]
        local = (distance - cumulative[index]) / span if span > 0 else 0.0
        return points[index] + (points[index + 1] - points[index]) * local
