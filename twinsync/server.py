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
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .routing import RoadNetwork
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
        network = RoadNetwork.load(DATA_DIR / "roads.geojson", self.world.frame)
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
