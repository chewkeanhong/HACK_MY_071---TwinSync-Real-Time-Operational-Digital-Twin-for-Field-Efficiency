"""The simulation core -- runnable with no browser attached.

Everything the dashboard shows is computed here. Keeping it headless means the logic can
be proved in a terminal before a single pixel is drawn, and it makes the A/B honest: the
same scenario, seed and world are pushed through both dispatch modes.

    python -m twinsync.sim --scenario data/scenario.json --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import numpy as np

from edge.detector import EdgeDetector
from edge.intelligence import IntelligenceLayer
from edge.telemetry import Fault, TowerTelemetry

from .coverage import CoverageEngine
from .dispatch import Crew, DispatchEngine
from .metrics import CO2_KG_PER_L, FUEL_L_PER_KM, Comparison, RunMetrics, collect
from .priority import assess
from .routing import RoadNetwork
from .weather import WeatherField
from .world import World

COVERAGE_CACHE = "coverage_cache.json"

# How often the flood scan re-walks the road graph. A storm cell moves a few metres a
# second; which segments are underwater does not change at the telemetry sample rate.
FLOOD_CHECK_PERIOD_S = 60.0


@dataclass
class SimState:
    """A snapshot of the world at one instant -- also the WebSocket payload."""

    t: float = 0.0
    tower_status: dict[str, str] = field(default_factory=dict)
    tower_digest: dict[str, dict] = field(default_factory=dict)
    dark_buildings: set[str] = field(default_factory=set)
    incidents: list[dict] = field(default_factory=list)
    crews: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class Simulation:
    """Drives towers, edge detectors, coverage and dispatch forward in time."""

    def __init__(self, world: World, coverage: CoverageEngine, network: RoadNetwork,
                 scenario: dict, *, smart: bool = True, seed: int = 42):
        self.world = world
        self.coverage = coverage
        self.network = network
        self.scenario = scenario
        self.smart = smart
        self.seed = seed

        self.sample_hz = float(scenario.get("sample_hz", 5.0))
        self.nominal_hz = float(scenario.get("nominal_sample_hz", 10.0))
        self.digest_period_s = float(scenario.get("digest_period_s", 10.0))
        self.baseline_delay_s = float(scenario.get("baseline_detection_delay_s", 600.0))
        self.sla_minutes = float(scenario.get("sla_minutes", 60.0))

        self.telemetry = {t.id: TowerTelemetry(t.id, seed=seed) for t in world.towers}
        self.detectors = {t.id: EdgeDetector(t.id, seed=seed) for t in world.towers}
        self.intelligence = IntelligenceLayer(world)
        self.faults: dict[str, Fault] = {}
        self._pending_faults = sorted(scenario.get("faults", []),
                                      key=lambda f: f["start_s"])

        crews = [
            Crew(id=c["id"], name=c["name"],
                 home_xy=np.array(world.frame.to_xy(c["lon"], c["lat"]), dtype=np.float64),
                 xy=np.array(world.frame.to_xy(c["lon"], c["lat"]), dtype=np.float64))
            for c in scenario.get("crews", [])
        ]
        self.dispatch = DispatchEngine(network, crews, smart=smart)

        self.t = 0.0
        self.failed_towers: set[str] = set()
        self.tower_status = {t.id: "healthy" for t in world.towers}
        self.state = SimState()
        self.events: list[dict] = []

        # Uplink accounting.
        self.samples_generated = 0
        self.events_uplinked = 0
        self._last_digest_at = -1e9
        self.encroachment_risk = self._build_encroachment_risk()

        self.weather = WeatherField.from_scenario(scenario, world.frame)
        # Each site's backhaul hop is modelled as the link to its nearest neighbour,
        # which is the usual shape of a chained urban aggregation network. Precomputed
        # because it never changes and it is needed on every telemetry sample.
        self.backhaul_peer = self._build_backhaul_peers()
        self.flooded_segments = 0
        self._last_flood_check = -1e9

        self._incident_by_tower: dict[str, str] = {}
        self._apply_congestion()

    def _build_encroachment_risk(self) -> dict[str, float]:
        """Static NDVI-style vegetation risk proxy for the prototype.

        In production this is expected to come from a Sentinel-2 pipeline. For v0.1 we use
        deterministic static values so UI/API plumbing can be demonstrated offline.
        """
        configured = self.scenario.get("encroachment_risk") or {}
        risk: dict[str, float] = {}
        for tower in self.world.towers:
            if tower.id in configured:
                value = float(configured[tower.id])
            else:
                seed = int(sha1(tower.id.encode("utf-8")).hexdigest()[:8], 16)
                value = 0.15 + ((seed % 70) / 100.0)
            risk[tower.id] = min(1.0, max(0.0, value))
        return risk

    def _build_backhaul_peers(self) -> dict[str, str]:
        """Nearest-neighbour backhaul topology, one hop per site."""
        peers: dict[str, str] = {}
        for tower in self.world.towers:
            others = [t for t in self.world.towers if t.id != tower.id]
            if not others:
                continue
            nearest = min(others, key=lambda t: float(np.hypot(*(t.xy - tower.xy))))
            peers[tower.id] = nearest.id
        return peers

    def weather_at(self, tower_id: str) -> dict:
        """Environmental conditions over one site, including its backhaul hop."""
        tower = self.world.tower(tower_id)
        conditions = self.weather.at(float(tower.xy[0]), float(tower.xy[1]), self.t)

        peer_id = self.backhaul_peer.get(tower_id)
        if peer_id is not None:
            capacity, fade = self.weather.backhaul_capacity(
                tower.xy, self.world.tower(peer_id).xy, self.t)
            conditions["backhaul_capacity"] = round(capacity, 4)
            conditions["backhaul_fade_db"] = round(fade, 2)
            conditions["backhaul_peer"] = peer_id
        conditions["encroachment_risk"] = self.encroachment_risk.get(tower_id, 0.4)
        return conditions

    # -- setup -----------------------------------------------------------

    def _apply_congestion(self) -> None:
        for entry in self.scenario.get("congestion", []):
            near = None
            if entry.get("near_tower"):
                near = self.world.tower(entry["near_tower"]).xy
            count = self.network.set_congestion(
                float(entry["factor"]),
                road_name=entry.get("road_name"),
                highway=entry.get("highway"),
                near=near,
                radius_m=float(entry.get("radius_m", 300.0)),
            )
            print(f"  congestion x{entry['factor']} applied to {count} road segments")

    # -- stepping --------------------------------------------------------

    def _release_faults(self) -> None:
        while self._pending_faults and self._pending_faults[0]["start_s"] <= self.t:
            spec = self._pending_faults.pop(0)
            self.faults[spec["tower"]] = Fault(
                tower_id=spec["tower"],
                profile=spec["profile"],
                start_s=float(spec["start_s"]),
            )
            self._log(f"fault injected at {spec['tower']} ({spec['profile']})")

    def _log(self, message: str) -> None:
        # The index lets a client append only what it has not already shown. Without it
        # the log is re-sent in full several times a second and every line repeats.
        self.events.append({"i": len(self.events), "t": round(self.t, 1),
                            "message": message})

    def _on_state_change(self, tower_id: str, state: str, reasons: list[str]) -> None:
        """The edge has changed its mind about a tower. React."""
        self.tower_status[tower_id] = state
        self.events_uplinked += 1

        fault = self.faults.get(tower_id)
        latency = (self.t - fault.start_s) if fault else 0.0

        if state in {"degraded", "down"}:
            if tower_id in self._incident_by_tower:
                return
            self.failed_towers.add(tower_id)
            impact = assess(self.coverage, self.failed_towers)
            incident = self.dispatch.report(self.t, tower_id, state, impact,
                                            self.world.tower(tower_id).xy,
                                            fault_started_at=fault.start_s if fault else self.t)
            profile = fault.profile if fault else "unknown"
            localisation = self.intelligence.localise(tower_id, profile, self.t)
            risk = self.intelligence.score_risk(
                tower_id,
                severity=state,
                subscribers=impact.subscribers,
                critical_count=impact.critical_count,
                minutes_open=0.0,
                sla_minutes=self.sla_minutes,
                weather=self.weather_at(tower_id),
                now_s=self.t,
            )
            incident.ai_cluster_id = localisation.cluster_id
            incident.ai_cluster_members = localisation.members
            incident.ai_cluster_noise = localisation.is_noise
            incident.ai_cluster_span_m = localisation.span_m
            incident.ai_cluster_span_s = localisation.span_s
            incident.ai_localised_at = self.t
            incident.ai_risk_score = risk.score
            incident.ai_risk_band = risk.band
            incident.ai_risk_factors = risk.top_factors
            incident.ai_model_source = self.intelligence.model_source
            self._incident_by_tower[tower_id] = incident.id
            self._log(f"EDGE {tower_id} -> {state} after {latency:.1f}s "
                      f"({'; '.join(reasons) or 'threshold'})")
            self._log(f"IMPACT {incident.id}: {impact.subscribers:,} subscribers, "
                      f"{len(impact.dark_buildings)} buildings dark -- a 2D coverage "
                      f"model reports {len(impact.dark_2d)}, missing "
                      f"{impact.missed_subscribers:,} of them")
            self._log(f"LOCALISE {incident.id}: {localisation.cluster_id} -- "
                      f"{localisation.describe()}")
            self._log(f"RISK {incident.id}: {risk.describe()} [{risk.model}]")
            self.dispatch.assign(self.t, incident)

    def _update_flooding(self) -> None:
        """Reprice flooded roads. Checked periodically, not every tick.

        A storm cell moves ~5 m/s, so which segments are under it changes on a scale of
        minutes; re-scanning 30k edges at the sample rate would dominate the run for no
        additional fidelity.
        """
        if not self.weather.any_cells:
            return
        if self.t - self._last_flood_check < FLOOD_CHECK_PERIOD_S:
            return
        self._last_flood_check = self.t

        previous = self.flooded_segments
        self.flooded_segments = self.weather.flooded_segments(
            self.network, self.world.terrain, self.t)
        if self.flooded_segments and not previous:
            self._log(f"WEATHER flooding on {self.flooded_segments} road segments -- "
                      "routing repriced")
        elif previous and not self.flooded_segments:
            self._log("WEATHER floodwater receded, roads back to normal")

    def step(self, dt: float) -> None:
        self.t += dt
        self._release_faults()
        self._update_flooding()

        emit_digest = (self.t - self._last_digest_at) >= self.digest_period_s
        if emit_digest:
            self._last_digest_at = self.t

        for tower in self.world.towers:
            fault = self.faults.get(tower.id)
            conditions = (self.weather_at(tower.id)
                          if self.weather.any_cells else None)
            sample = self.telemetry[tower.id].sample(self.t, fault, conditions)
            # Raw-stream accounting uses the nominal rate the hardware would produce,
            # not the coarser step this simulation runs at.
            self.samples_generated += max(1, int(round(self.nominal_hz * dt)))

            if self.smart:
                verdict = self.detectors[tower.id].observe(sample)
                if verdict.changed:
                    self._on_state_change(tower.id, verdict.state, verdict.reasons)
            else:
                # No edge inference: the fault surfaces only after the complaint delay.
                if fault and not fault.resolved_at:
                    elapsed = self.t - fault.start_s
                    if (elapsed >= self.baseline_delay_s
                            and tower.id not in self._incident_by_tower):
                        severity = "down" if fault.profile == "power_failure" else "degraded"
                        self.failed_towers.add(tower.id)
                        impact = assess(self.coverage, self.failed_towers)
                        incident = self.dispatch.report(self.t, tower.id, severity,
                                                        impact, tower.xy,
                                                        fault_started_at=fault.start_s)
                        self._incident_by_tower[tower.id] = incident.id
                        self._log(f"TICKET {tower.id} reported by customers after "
                                  f"{elapsed / 60:.1f} min")
                        self.dispatch.assign(self.t, incident)

            if emit_digest:
                digest = self.telemetry[tower.id].digest(sample)
                digest["encroachment_risk"] = round(
                    100.0 * self.encroachment_risk.get(tower.id, 0.0), 1
                )
                digest["encroachment_source"] = "sentinel2-ndvi-simulated-v0.1"
                if conditions:
                    digest["rainfall_mm_hr"] = conditions["rainfall_mm_hr"]
                    digest["backhaul_fade_db"] = conditions.get("backhaul_fade_db", 0.0)
                    digest["backhaul_capacity"] = conditions.get("backhaul_capacity", 1.0)
                self.state.tower_digest[tower.id] = digest
                self.events_uplinked += 1

        self.dispatch.tick(self.t, dt)
        self._settle_resolved()

    def _settle_resolved(self) -> None:
        """Clear faults whose crews have finished, and recompute what is still dark."""
        for tower_id, incident_id in list(self._incident_by_tower.items()):
            incident = self.dispatch.incidents[incident_id]
            if not incident.resolved:
                continue
            fault = self.faults.get(tower_id)
            if fault and fault.resolved_at is None:
                fault.resolved_at = incident.resolved_at
            if tower_id in self.failed_towers:
                self.failed_towers.discard(tower_id)
                self.tower_status[tower_id] = "healthy"
                self.detectors[tower_id].state = "healthy"
                # Drop the alarm from the clustering window too, or a resolved site
                # keeps pulling later, unrelated faults into its cluster.
                self.intelligence.release(tower_id)
                self._log(f"{tower_id} restored to service")
            self._incident_by_tower.pop(tower_id, None)

        self.state.dark_buildings = self.coverage.outage(self.failed_towers)

    # -- snapshot --------------------------------------------------------

    # Frames carry only a tail of the event log; clients skip what they already have.
    EVENT_TAIL = 40

    def snapshot(self) -> dict:
        """Everything the dashboard needs for one frame.

        Deliberately excludes the building geometry, which is static and fetched once
        over HTTP -- pushing a megabyte of polygons at 4 Hz would be absurd.
        """
        frame = self.world.frame

        crews = []
        for crew in self.dispatch.crews:
            position = self.dispatch.position_of(crew)
            lon, lat = frame.to_lonlat(position[0], position[1])
            entry = {
                "id": crew.id,
                "name": crew.name,
                "lon": round(float(lon), 7),
                "lat": round(float(lat), 7),
                "status": crew.status,
                "eta_min": round(crew.eta_s / 60.0, 1),
                "queue": list(crew.queue),
                "trips": crew.trips,
            }
            if crew.route is not None:
                entry["route"] = crew.route.to_lonlat(frame)
            crews.append(entry)

        incidents = []
        for incident in self.dispatch.incidents.values():
            if incident.resolved:
                continue
            incidents.append({
                "id": incident.id,
                "tower": incident.tower_id,
                "severity": incident.severity,
                "priority": round(incident.priority, 2),
                "assigned_to": incident.assigned_to,
                "subscribers": incident.impact.subscribers,
                "buildings_dark": len(incident.impact.dark_buildings),
                "buildings_2d": incident.impact.naive_buildings,
                "buildings_dark_2d": len(incident.impact.dark_2d),
                "missed_by_2d": incident.impact.missed_by_2d,
                "crew_2d": incident.crew_2d,
                "crew_2d_minutes": (round(incident.crew_2d_minutes, 1)
                                    if incident.crew_2d_minutes is not None else None),
                "assigned_minutes": (round(incident.assigned_minutes, 1)
                                     if incident.assigned_minutes is not None else None),
                "critical_sites": incident.impact.critical_sites,
                "minutes_open": round((self.t - incident.detected_at) / 60.0, 1),
                "sla_minutes_left": round(
                    self.sla_minutes - (self.t - incident.detected_at) / 60.0, 1),
                "ai_cluster_id": incident.ai_cluster_id,
                "ai_cluster_members": incident.ai_cluster_members,
                "ai_cluster_noise": incident.ai_cluster_noise,
                "ai_cluster_span_m": round(incident.ai_cluster_span_m, 1),
                "ai_cluster_span_s": round(incident.ai_cluster_span_s, 1),
                "ai_risk_score": incident.ai_risk_score,
                "ai_risk_band": incident.ai_risk_band,
                "ai_risk_factors": incident.ai_risk_factors,
                "ai_model_source": incident.ai_model_source,
            })
        incidents.sort(key=lambda i: -i["priority"])

        distance_km = sum(c.distance_m for c in self.dispatch.crews) / 1000.0

        # Storm cells in lon/lat so the client can draw them without knowing about the
        # local metre frame.
        cells = []
        for cell in self.weather.active_cells(self.t):
            lon, lat = frame.to_lonlat(cell["x"], cell["y"])
            cells.append({
                "id": cell["id"],
                "lon": round(float(lon), 7),
                "lat": round(float(lat), 7),
                "radius_m": cell["radius_m"],
                "intensity": cell["intensity"],
                "rain_mm_hr": cell["rain_mm_hr"],
            })

        flooded = []
        if self.flooded_segments:
            seen = set()
            for a, b, data in self.network.graph.edges(data=True):
                if not data.get("flooded"):
                    continue
                key = (min(a, b), max(a, b))
                if key in seen:
                    continue
                seen.add(key)
                ax, ay = self.network.node_xy[a]
                bx, by = self.network.node_xy[b]
                lon0, lat0 = frame.to_lonlat(ax, ay)
                lon1, lat1 = frame.to_lonlat(bx, by)
                flooded.append([[round(float(lon0), 7), round(float(lat0), 7)],
                                [round(float(lon1), 7), round(float(lat1), 7)]])

        return {
            "t": round(self.t, 1),
            "tower_status": dict(self.tower_status),
            "tower_digest": dict(self.state.tower_digest),
            "weather": {
                "cells": cells,
                "flooded_segments": self.flooded_segments,
                "flooded_paths": flooded,
                "profile": self.weather.profile,
            },
            "dark_buildings": sorted(self.state.dark_buildings),
            # What a fair 2D model concludes is dark, and the raw circle it would draw.
            "dark_buildings_2d": sorted(self.coverage.outage_2d(self.failed_towers)),
            "naive_radius": sorted(self.coverage.naive_radius(self.failed_towers)),
            "incidents": incidents,
            "crews": crews,
            "events": self.events[-self.EVENT_TAIL:],
            "event_count": len(self.events),
            "uplink": {
                "raw_bytes": self.samples_generated * 120,
                "sent_bytes": self.events_uplinked * 180,
                "events": self.events_uplinked,
            },
            # Fleet effort so far, from the routes actually driven. Same constants the
            # results table uses, so the dashboard and the slide cannot disagree.
            "fleet": {
                "truck_rolls": sum(c.trips for c in self.dispatch.crews),
                "travel_km": round(distance_km, 2),
                "fuel_litres": round(distance_km * FUEL_L_PER_KM, 2),
                "co2_kg": round(distance_km * FUEL_L_PER_KM * CO2_KG_PER_L, 2),
            },
        }

    # -- running ---------------------------------------------------------

    def run(self, duration_s: float, *, dt: float | None = None,
            unassigned_retry_s: float = 60.0) -> RunMetrics:
        dt = dt if dt is not None else 1.0 / self.sample_hz
        steps = int(duration_s / dt)
        next_retry = unassigned_retry_s

        for _ in range(steps):
            self.step(dt)
            # Incidents parked because every crew was busy need another look.
            if self.t >= next_retry:
                next_retry = self.t + unassigned_retry_s
                for incident in self.dispatch.open_incidents:
                    if incident.assigned_to is None:
                        self.dispatch.assign(self.t, incident)

        label = "TwinSync" if self.smart else "today"
        return collect(label, self.dispatch, sla_minutes=self.sla_minutes,
                       samples_generated=self.samples_generated,
                       events_uplinked=self.events_uplinked,
                       elapsed_seconds=self.t,
                       total_subscribers=self.world.total_subscribers)


# ------------------------------------------------------------------ loading


def load_all(data_dir: Path, *, verbose: bool = True) -> tuple[World, CoverageEngine,
                                                              RoadNetwork]:
    world = World.load(data_dir, require_towers=True)
    if verbose:
        print(f"world:   {world.summary()}")

    coverage = CoverageEngine(world)
    cache = data_dir / COVERAGE_CACHE
    if coverage.load(cache):
        if verbose:
            print(f"coverage: loaded from {cache}")
    else:
        if verbose:
            print("coverage: computing 3D line-of-sight...")
        coverage.compute(verbose=verbose)
        coverage.save(cache)

    network = RoadNetwork.load(data_dir / "roads.geojson", world.frame)
    if verbose:
        print(f"network: {network.summary()}")
    return world, coverage, network


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the TwinSync simulation headless.")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--scenario", default="data/scenario.json", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", dest="json_out", type=Path, default=None,
                        help="write the comparison as JSON")
    args = parser.parse_args(argv)

    scenario = json.loads(args.scenario.read_text(encoding="utf-8"))
    duration = args.duration or float(scenario.get("duration_s", 3600.0))
    verbose = not args.quiet

    world, coverage, network = load_all(args.data, verbose=verbose)

    print(f"\nscenario: {scenario.get('name')}")
    print(f"  {duration:.0f}s simulated, seed {args.seed}, "
          f"baseline detection delay {scenario['baseline_detection_delay_s'] / 60:.0f} min")

    results = {}
    for smart in (False, True):
        arm = "TwinSync" if smart else "today"
        print(f"\n{'=' * 64}\n{arm}\n{'=' * 64}")
        # A fresh network per arm so congestion state cannot leak between them.
        arm_network = RoadNetwork.load(args.data / "roads.geojson", world.frame)
        sim = Simulation(world, coverage, arm_network, scenario,
                         smart=smart, seed=args.seed)
        results[smart] = sim.run(duration)

        for t, message in sim.dispatch.log:
            print(f"  [{t / 60:6.1f} min] {message}")
        if verbose:
            for event in sim.events:
                print(f"  [{event['t'] / 60:6.1f} min] {event['message']}")

    comparison = Comparison(baseline=results[False], twinsync=results[True])
    print(f"\n{'=' * 64}\nRESULTS\n{'=' * 64}")
    print(comparison.render())

    if args.json_out:
        args.json_out.write_text(json.dumps(comparison.as_dict(), indent=2),
                                 encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
