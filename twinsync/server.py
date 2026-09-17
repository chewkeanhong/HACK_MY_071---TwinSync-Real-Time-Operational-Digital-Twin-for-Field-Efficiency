"""FastAPI server: owns the clock, streams state, serves the dashboard.

The browser is a pure renderer. It never simulates anything, so there is no way for the
picture to drift out of step with the model behind it -- the server computes a frame and
pushes it, and every connected client sees the same thing.

Static geometry (2000 building footprints) goes over HTTP once and is cached. Only the
mutable state -- tower health, dark buildings, crew positions, incidents -- rides the
WebSocket, at a few frames a second.

    python -m uvicorn twinsync.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import checkpoints
from .routing import DRY, RoadNetwork
from .sim import Simulation, load_all

DATA_DIR = Path(os.environ.get("TWINSYNC_DATA", "data"))
WEB_DIR = Path(__file__).resolve().parents[1] / "web"

# Wall-clock seconds between pushed frames.
FRAME_PERIOD_S = 0.25

# How much faster than real time the scenario plays. An hour-long incident timeline has
# to fit inside a five-minute pitch.
DEFAULT_TIME_SCALE = 12.0


class Engine:
    """Holds the world and the running simulation, and drives it forward."""

    def __init__(self) -> None:
        self.world = None
        self.coverage = None
        self.network = None
        self.scenario: dict = {}
        self.sim: Simulation | None = None
        self.time_scale = DEFAULT_TIME_SCALE
        self.paused = False
        self.clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    def load(self) -> None:
        self.world, self.coverage, self.network = load_all(DATA_DIR)
        self.scenario = json.loads(
            (DATA_DIR / "scenario.json").read_text(encoding="utf-8")
        )
        self.reset()

    def reset(self) -> None:
        # A fresh road network per run so congestion from a previous run cannot persist.
        network = RoadNetwork.load(DATA_DIR / "roads.geojson", self.world.frame,
                                   terrain=self.world.terrain)
        self.sim = Simulation(self.world, self.coverage, network, self.scenario,
                              smart=True, seed=int(self.scenario.get("seed", 42)))
        self.paused = False

    def static_payload(self) -> dict:
        """Everything that never changes, fetched once by the client."""
        buildings = json.loads((DATA_DIR / "buildings.geojson").read_text(encoding="utf-8"))
        towers = json.loads((DATA_DIR / "towers.geojson").read_text(encoding="utf-8"))

        lons = [b.centroid_lonlat[0] for b in self.world.buildings]
        lats = [b.centroid_lonlat[1] for b in self.world.buildings]

        return {
            "buildings": buildings,
            "towers": towers,
            "subscribers": {b.id: b.subscribers for b in self.world.buildings},
            "critical": [b.id for b in self.world.buildings if b.critical],
            "centre": {"lon": sum(lons) / len(lons), "lat": sum(lats) / len(lats)},
            "scenario": self.scenario,
            "total_subscribers": self.world.total_subscribers,
            # Transport topology is static, so it belongs in the one-off payload
            # rather than in a frame the socket repeats four times a second.
            "asset_graph": self.world.asset_graph.to_dict(self.world.frame,
                                                          self.world.towers),
        }

    async def broadcast(self, payload: dict) -> None:
        if not self.clients:
            return
        message = json.dumps(payload)
        dead = []
        for client in list(self.clients):
            try:
                await client.send_text(message)
            except Exception:
                dead.append(client)
        for client in dead:
            self.clients.discard(client)

    async def run_loop(self) -> None:
        """Advance the simulation in wall-clock time and push frames."""
        while True:
            await asyncio.sleep(FRAME_PERIOD_S)
            if self.paused or self.sim is None:
                continue
            async with self._lock:
                # One frame of wall time is time_scale seconds of simulated time,
                # stepped at the scenario's sample rate so detector behaviour is
                # identical to the headless run.
                dt = 1.0 / self.sim.sample_hz
                simulated = FRAME_PERIOD_S * self.time_scale
                for _ in range(max(1, int(round(simulated / dt)))):
                    self.sim.step(dt)
                payload = self.sim.snapshot()
            await self.broadcast(payload)


engine = Engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load()
    task = asyncio.create_task(engine.run_loop())
    yield
    task.cancel()


app = FastAPI(title="TwinSync", lifespan=lifespan)


@app.middleware("http")
async def always_revalidate(request, call_next):
    """Make the browser check with the server before reusing anything it has cached.

    With no Cache-Control header a browser guesses how long a response stays fresh from
    its Last-Modified date -- and a file that sat unchanged for days is judged fresh for
    hours. So after editing the dashboard, a normal reload served the *old* app.js from
    cache without asking: the guided demo's new step buttons were on disk and on the wire,
    and invisible on screen. On a demo laptop that is a silent failure minutes before
    presenting.

    `no-cache` does not disable caching. The browser still keeps its copy; it just asks
    first, and an unchanged file comes back as a cheap 304 -- including the vendored
    deck.gl bundle -- so an unmodified dashboard loads as fast as before.
    """
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.get("/api/world")
async def get_world():
    return JSONResponse(engine.static_payload())


@app.get("/api/roads")
async def get_roads():
    """Road geometry, drawn by the client instead of a basemap.

    Rendering our own roads rather than fetching map tiles is what makes the dashboard
    work with no network at all -- the demo cannot be broken by conference wifi.
    """
    return FileResponse(DATA_DIR / "roads.geojson", media_type="application/json")


@app.get("/api/state")
async def get_state():
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    return JSONResponse(engine.sim.snapshot())


@app.post("/api/control/{action}")
async def control(action: str, factor: float | None = None):
    """Demo controls: pause, resume, reset, speed."""
    if action == "pause":
        engine.paused = True
    elif action == "resume":
        engine.paused = False
    elif action == "reset":
        async with engine._lock:
            engine.reset()
    elif action == "speed" and factor:
        engine.time_scale = max(1.0, min(60.0, float(factor)))
    else:
        return JSONResponse({"error": f"unknown action {action}"}, status_code=400)
    return {"ok": True, "paused": engine.paused, "time_scale": engine.time_scale}


@app.post("/api/fault/{tower_id}")
async def inject_fault(tower_id: str, profile: str = "amplifier_degradation"):
    """Trigger a fault by hand -- the fallback if the scripted timeline misbehaves live."""
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    from edge.telemetry import FAULT_PROFILES, Fault

    if profile not in FAULT_PROFILES:
        return JSONResponse({"error": f"unknown profile {profile}"}, status_code=400)
    async with engine._lock:
        engine.sim.faults[tower_id] = Fault(tower_id=tower_id, profile=profile,
                                            start_s=engine.sim.t)
    return {"ok": True, "tower": tower_id, "profile": profile}


@app.post("/api/storm")
async def spawn_storm(radius_m: float = 1600.0, peak_mm_hr: float = 95.0,
                      duration_s: float = 1200.0, drift_bearing_deg: float = 225.0,
                      drift_kmh: float = 16.0):
    """Drop a monsoon cell over the AOI, now.

    Spawned upwind of the scene centre so the drift carries it across, which is what
    makes the effects visible: backhaul rain fade, then flooding on the low-lying roads
    the DEM identifies, then dispatch rerouting around them.
    """
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)

    from .weather import StormCell

    sim = engine.sim
    # Enter from the upwind edge: the cell travels along drift_bearing, so start it
    # 2 km back along that heading from the centre of the world.
    bearing = math.radians(drift_bearing_deg)
    centre = np.array([b.centroid_xy for b in sim.world.buildings]).mean(axis=0)
    start = centre - np.array([math.sin(bearing), math.cos(bearing)]) * 2000.0

    async with engine._lock:
        sim.weather.cells.append(StormCell(
            x=float(start[0]), y=float(start[1]),
            radius_m=float(radius_m), peak_mm_hr=float(peak_mm_hr),
            start_s=sim.t, duration_s=float(duration_s),
            drift_bearing_deg=float(drift_bearing_deg), drift_kmh=float(drift_kmh),
        ))
        # Force the next tick to re-scan rather than wait out the flood check period.
        sim._last_flood_check = -1e9
        sim._log(f"WEATHER operator injected a monsoon cell "
                 f"({peak_mm_hr:.0f} mm/hr, {radius_m / 1000:.1f} km)")

    return {"ok": True, "cells": len(sim.weather.cells), "peak_mm_hr": peak_mm_hr}


@app.post("/api/flood")
async def set_flood(surcharge_m: float = 0.6, graded: bool = True):
    """Raise standing water to ``surcharge_m`` above the drainage line.

    The storm route to flooding is realistic but slow: a cell has to drift in before
    anything gets wet. This drives the water level directly, which is what makes the
    flood-adaptive routing demonstrable on demand rather than on the weather's
    schedule. Pass a negative surcharge to drain the roads again.
    """
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)

    sim = engine.sim
    network = sim.network
    if network.flood_source == "no-dem":
        return JSONResponse(
            {"error": "flood routing needs a DEM -- run scripts/fetch_dem.py"},
            status_code=409)

    async with engine._lock:
        draining = surcharge_m < 0.0
        level = DRY if draining else network.surcharge_to_level(surcharge_m)
        wet = network.set_water_level(level, graded=graded)
        sim.flooded_segments = wet
        # Hold the level against the periodic rain scan, which would otherwise drain it
        # at the next check. A live storm cell still takes over -- real weather beats a
        # slider -- but an empty sky must not.
        sim.operator_water_level = None if draining else level
        sim._last_flood_check = sim.t
        sim._log("FLOOD operator drained the roads" if draining else
                 f"FLOOD operator set water to +{surcharge_m:.2f} m above the drainage "
                 f"line -- {wet} road segments under water")

    # A drained network sits at DRY, which is -inf and not representable in JSON.
    return {"ok": True, "surcharge_m": surcharge_m,
            "water_level": None if draining else round(level, 2),
            "flood_datum": round(network.flood_datum, 2), "flooded_segments": wet,
            "graded": graded}


@app.post("/api/blackout/{tower_id}")
async def blackout(tower_id: str, minutes: float | None = None):
    """Cut mains to a site and start its battery countdown.

    The cascade's trigger. With realistic autonomy nothing goes dark inside a demo, so
    pass a short ``minutes`` to watch the blackout and the wave behind it actually
    happen.
    """
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    try:
        async with engine._lock:
            autonomy = engine.sim.put_on_battery(tower_id, minutes=minutes)
    except KeyError:
        return JSONResponse({"error": f"unknown tower {tower_id}"}, status_code=400)
    return {"ok": True, "tower": tower_id, "autonomy_s": autonomy}


@app.get("/api/cascade/{tower_id}")
async def cascade(tower_id: str):
    """What would go dark if this site failed, without actually failing it."""
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    sim = engine.sim
    if tower_id not in sim.world.asset_graph.graph:
        return JSONResponse({"error": f"unknown tower {tower_id}"}, status_code=404)

    impact = sim.world.calculate_cascade_impact(
        tower_id,
        already_failed=sim.failed_towers,
        on_battery=sim.battery_started_at.keys(),
        battery_remaining_s=sim.battery_remaining_s(),
    )
    subscribers = sim.coverage.subscribers_affected(
        sim.coverage.outage(set(impact.dark)))
    return {
        "tower": tower_id,
        "tier": sim.world.asset_graph.tier.get(tower_id),
        "dark": list(impact.dark),
        "isolated": list(impact.cascaded),
        "at_risk": list(impact.at_risk),
        "unprotected": list(impact.unprotected),
        "depth": impact.depth,
        "subscribers_at_risk": int(subscribers),
        "restoration_order": sim.world.asset_graph.restoration_order(impact.dark),
        "summary": impact.describe(),
    }


@app.get("/api/route")
async def resilient_route(from_lon: float, from_lat: float,
                          to_lon: float, to_lat: float,
                          surcharge_m: float | None = None):
    """A crew route that avoids floodwater, as a GeoJSON Feature.

    ``surcharge_m`` overrides the live water level for a what-if ("what would this
    trip look like under another half metre"); omitted, it answers against whatever
    the roads are actually carrying right now.
    """
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)

    network = engine.sim.network
    if surcharge_m is None:
        level = network.water_level
    else:
        level = network.surcharge_to_level(surcharge_m)

    answer = network.get_resilient_route((from_lon, from_lat), (to_lon, to_lat), level)
    return answer.to_geojson(network.frame)


@app.get("/api/models")
async def get_models():
    """What is actually running, with artifact hashes. The judge-facing 'prove it'.

    Reads the metadata the training scripts wrote, so this cannot drift from the models
    on disk: if an artifact is missing, this says so rather than reporting the claim.
    """
    models_dir = Path(__file__).resolve().parents[1] / "models"
    entries = []

    for name, filename in (("edge anomaly autoencoder", "edge_anomaly_meta.json"),
                           ("7-day failure risk", "risk_meta.json")):
        path = models_dir / filename
        if not path.exists():
            entries.append({"name": name, "status": "missing",
                            "note": "run the matching script in scripts/"})
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        entries.append({
            "name": name,
            "status": "loaded",
            "model": meta.get("model"),
            "created": meta.get("created"),
            "trained_on": meta.get("trained_on") or meta.get("training_data"),
            "artifact": meta.get("shipped_artifact", "risk_lgbm.txt"),
            "sha256": meta.get("sha256") or meta.get("sha256_fp32"),
            "metrics": meta.get("metrics") or {
                "healthy_false_positive_rate": meta.get("healthy_false_positive_rate"),
                "threshold": meta.get("threshold"),
            },
        })

    terrain = engine.world.terrain if engine.world is not None else None
    return {
        "models": entries,
        # Not models, but the same question -- what is real and what is a stand-in.
        "algorithms": [
            {"name": "ST-DBSCAN fault localisation", "model": "st-dbscan-v1",
             "note": "algorithm, not a trained model -- no artifact to ship"},
            {"name": "3D coverage", "model": "fresnel-raycast-v1",
             "note": "deterministic ray-casting, 60% first-Fresnel clearance"},
        ],
        "data": {
            "terrain": terrain.meta.describe() if terrain else "not loaded",
            "terrain_is_real": bool(terrain and terrain.meta.is_real),
            "encroachment": (engine.world.encroachment.describe()
                             if engine.world is not None else "not loaded"),
            "encroachment_is_real": bool(engine.world is not None
                                         and engine.world.encroachment.is_real),
        },
        "degraded": engine.sim.intelligence.degraded if engine.sim else None,
    }


@app.get("/api/metrics")
async def get_metrics(sites: int | None = None,
                      incidents_per_site: float | None = None):
    """Live KPI rollup for the dashboard tiles, plus the A/B saving.

    The live server runs one arm, so the comparison cannot be measured here -- it comes
    from `data/results.json`, written by the headless run that plays the identical
    scenario through both dispatch modes. Only the *projection* is recomputed, which is
    what `sites` and `incidents_per_site` are for: the two multipliers are assumptions
    rather than findings, so a judge who disagrees with 2,000 sites can say so and watch
    the number move instead of being told to trust it.
    """
    if engine.sim is None:
        return JSONResponse({"error": "not ready"}, status_code=503)
    from .metrics import (DEFAULT_INCIDENT_RATE, DEFAULT_SITES, collect,
                          per_incident_from_results, project_annual)

    sim = engine.sim
    run = collect("live", sim.dispatch, sla_minutes=sim.sla_minutes,
                  samples_generated=sim.samples_generated,
                  events_uplinked=sim.events_uplinked,
                  elapsed_seconds=sim.t,
                  total_subscribers=sim.world.total_subscribers)
    payload = run.as_dict()

    results_path = DATA_DIR / "results.json"
    if results_path.exists():
        saved = json.loads(results_path.read_text(encoding="utf-8"))
        unit = per_incident_from_results(saved)
        payload["ab"] = {
            "source": "data/results.json",
            "note": ("measured A/B: identical scenario, seed, world and crews, run "
                     "through baseline dispatch and TwinSync dispatch"),
            "truck_rolls_avoided": saved.get("truck_rolls_avoided"),
            "cost_saved_myr": saved.get("cost_saved_myr"),
            "km_saved": saved.get("km_saved"),
            "mttr_improvement_pct": saved.get("mttr_improvement_pct"),
            "mttd_improvement_pct": saved.get("mttd_improvement_pct"),
            # Both sides of the detection clock, not just the delta. The MTTD tile
            # renders "10.0 min -> 2.4 s", and a percentage alone cannot say that.
            "mttd_baseline_minutes": saved.get("baseline", {}).get(
                "mean_detection_minutes"),
            "mttd_twinsync_minutes": saved.get("twinsync", {}).get(
                "mean_detection_minutes"),
            "per_incident": {k: round(v, 3) for k, v in unit.items()},
            "annualised": project_annual(
                unit,
                sites=max(1, int(sites if sites is not None else DEFAULT_SITES)),
                incidents_per_site_per_year=float(
                    incidents_per_site if incidents_per_site is not None
                    else DEFAULT_INCIDENT_RATE),
            ),
        }
    return payload


@app.get("/api/terrain")
async def get_terrain():
    """The DEM grid, so the client can shade the ground it is drawing on."""
    path = DATA_DIR / "terrain.json"
    if not path.exists():
        return JSONResponse({"error": "no terrain baked"}, status_code=404)
    return FileResponse(path, media_type="application/json")


def _demo_beats() -> list[dict]:
    """The beat track, or an empty list if this repo has no demo baked."""
    path = DATA_DIR / "demo.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("beats", [])
    except (OSError, json.JSONDecodeError):
        return []


@app.get("/api/demo")
async def get_demo():
    """The guided-demo beat track.

    Optional: a repo without one still runs, the button just stays disabled. The beats
    only narrate -- they never inject a fault or a storm -- so the guided run is the
    scripted scenario every time, which is the point of having it.
    """
    path = DATA_DIR / "demo.json"
    if not path.exists():
        return JSONResponse({"error": "no demo track"}, status_code=404)
    return FileResponse(path, media_type="application/json")


@app.get("/api/demo/checkpoints")
async def demo_checkpoints():
    """Which beats the presenter can jump to, and why not if they cannot.

    The client asks once at load so the Prev/Next buttons can say what is wrong instead
    of failing silently: a missing or stale recording is a five-minute fix with
    `scripts/bake_checkpoints.py`, but only if somebody is told before they are on stage.
    """
    beats = _demo_beats()
    if not beats:
        return {"available": False, "reason": "no demo track", "beats": 0}
    manifest = checkpoints.load_manifest(DATA_DIR, beats)
    return {"available": manifest.usable, "reason": manifest.reason,
            "beats": len(manifest.beats) if manifest.usable else 0}


@app.post("/api/demo/seek/{index}")
async def demo_seek(index: int):
    """Move the scenario clock to a beat, by restoring its recorded state.

    Replaying to the instant would take minutes (~80 ms per 0.2 s step), so the run is
    recorded ahead of time and this is a read. The restored simulation is identical to
    having let the demo run there, and it keeps running from that point at whatever speed
    the clock is set to.
    """
    beats = _demo_beats()
    if not beats:
        return JSONResponse({"error": "no demo track"}, status_code=404)
    manifest = checkpoints.load_manifest(DATA_DIR, beats)
    if not manifest.usable:
        return JSONResponse({"error": manifest.reason}, status_code=409)
    if not 0 <= index < len(manifest.beats):
        return JSONResponse({"error": f"no beat {index}"}, status_code=404)

    async with engine._lock:
        if engine.sim is None:
            return JSONResponse({"error": "not ready"}, status_code=503)
        try:
            # The live simulation is only read from here, for the world, coverage and
            # model objects the recording deliberately left out.
            restored = checkpoints.load(manifest.path_for(index), engine.sim)
        except Exception as exc:                                  # noqa: BLE001
            return JSONResponse(
                {"error": f"could not restore beat {index}: {exc}"}, status_code=500)
        engine.sim = restored
        payload = engine.sim.snapshot()

    # Push the new state immediately rather than waiting for the next frame, so the
    # dashboard repaints on the beat instead of a quarter-second later.
    await engine.broadcast(payload)
    return {"ok": True, "beat": index, "t": payload["t"],
            "paused": engine.paused, "time_scale": engine.time_scale}


@app.get("/api/coverage/{tower_id}")
async def tower_coverage(tower_id: str):
    """The 2D-versus-3D comparison for one tower, for the on-screen toggle."""
    if engine.coverage is None or tower_id not in engine.coverage.by_tower:
        return JSONResponse({"error": "unknown tower"}, status_code=404)
    entry = engine.coverage.by_tower[tower_id]
    stats = engine.coverage.compare_2d_vs_3d(tower_id)
    return {
        **stats,
        "covered": sorted(entry.covered),
        "blocked": sorted(entry.blocked),
    }


@app.websocket("/ws")
async def websocket_endpoint(socket: WebSocket):
    await socket.accept()
    engine.clients.add(socket)
    try:
        if engine.sim is not None:
            await socket.send_text(json.dumps(engine.sim.snapshot()))
        while True:
            # The client never drives the simulation; this just keeps the socket open
            # and notices when it goes away.
            await socket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        engine.clients.discard(socket)


@app.get("/")
async def index():
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
