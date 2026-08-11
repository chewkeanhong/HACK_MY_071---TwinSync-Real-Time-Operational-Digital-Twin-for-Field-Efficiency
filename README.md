# TwinSync — Real-Time Operational Digital Twin for Field Efficiency

**ASEAN GeoAI Fusion 2026 · Theme: Efficiency · Kuala Lumpur CBD**

A live 3D digital twin of a telecom network. Simulated edge nodes on each tower run
anomaly inference locally and uplink *events, not telemetry*. The twin recomputes true
line-of-sight coverage against extruded buildings, scores who actually lost service, and
dispatches crews over the real street graph.

> A 2D map cannot tell you who lost signal, because signal is blocked in 3D — and that is
> why crews get sent to the wrong place.

---

## Quick start

```bash
pip install fastapi uvicorn websockets shapely networkx pytest
python -m uvicorn twinsync.server:app --port 8000
# open http://localhost:8000
```

Runs entirely offline. No map tiles, no CDN, no API keys — deck.gl is vendored and the
roads are drawn from our own GeoJSON.

**Headless, no browser needed:**
```bash
python -m twinsync.sim --scenario data/scenario.json --seed 42
pytest tests/ -q
```

---

## Prototype scope disclosure (v0.1)

This repository is explicit about what is production model output vs simulated output.

- **ST-DBSCAN fault localisation** and **LightGBM risk scoring** are currently
  **simulated** in [edge/intelligence.py](edge/intelligence.py). The simulator emits
  stable cluster IDs and risk bands so the API and UI integration are real, while model
  training artifacts (`.pkl`, `.onnx`) are deferred.
- Incidents expose these outputs through `ai_*` fields in `/api/state`, and the incident
  panel labels them as `simulated-v0.1`.
- **Copernicus DEM GLO-30 is not yet integrated** in this version. The current Z-axis
  reasoning comes from extruded building heights (OSM + imputed), not terrain raster
  elevation.
- **Edge ONNX/TFLite artifacts are not included** in this repository. The edge tier in
  [edge/detector.py](edge/detector.py) and [edge/telemetry.py](edge/telemetry.py) simulates
  what would run on Jetson Orin Nano / Raspberry Pi class hardware in deployment.

This keeps claims honest for a hackathon prototype while preserving a clean integration
point for swapping in real trained models and DEM-backed terrain later.

---

## Future work & scaling

- **Sentinel-2 L2A NDVI for vegetation encroachment** is part of the target architecture.
  For this 4-day prototype, encroachment risk is mocked as static per-tower variables in
  the simulation output (`tower_digest.encroachment_risk`) so dashboard and control logic
  can be validated without a raster processing pipeline.
- **SHAP explainability** is currently represented as simulated attribution text in tower
  tooltips when a tower is degraded/down. Human-in-the-loop approval workflow is on the
  UI roadmap for production dispatch.

---

## The four GeoAI components

| # | Component | What it does |
|---|---|---|
| 1 | **Height imputation** | Random forest fills the 83% of footprints OSM has no height for |
| 2 | **3D line-of-sight coverage** | Ray-casts antenna → facade against extruded buildings |
| 3 | **Edge anomaly detection** | EWMA + Mahalanobis at full rate, Isolation Forest on a duty cycle |
| 4 | **Impact-ranked dispatch** | A\* on travel time, trip batching, priority preemption |

---

## Measured results

The "today" column is not an assumption — it is the *same scenario, seed, world and
crews* run through a baseline dispatch mode (FIFO, nearest-by-distance, no batching, no
edge inference). It is a controlled experiment, not a marketing claim.

| metric | today | TwinSync |
|---|---|---|
| mean time to restore | 40.3 min | **35.7 min** (−11%) |
| mean detection time | 10.0 min | **0.05 min** |
| truck rolls | 4 | **3** |
| subscriber-minutes lost | 443,766 | **400,275** |
| edge uplink | 63.3 MB raw | **942 KB (−98.5%)** |

**The 11% is honest and modest**, because on-site repair time dominates and is identical
in both arms. The large win is detection (10 min → 3 s) and backhaul (−98.5%).

### The headline 3D number

With three towers down, a fair 2D coverage model — inside a failed circle, outside every
healthy one — reports **3 buildings dark, 3 subscribers**. True 3D line-of-sight says
**40 buildings, 6,703 subscribers**.

**The flat map misses 6,700 people who are genuinely off the air.** It does not raise a
false alarm; it says "the neighbouring cell has them" and is wrong, because a 200 m tower
stands between those buildings and the neighbouring cell.

The dashboard has three view modes, switchable by button or the `1` `2` `3` keys:

| key | mode | what it shows |
|---|---|---|
| `1` | **2D** | the flat coverage map as dispatch draws it today — every tower's circle, overlapping |
| `2` | **3D** | true line of sight against the extruded city (default) |
| `3` | **Compare** | both, side by side, off the same instant of the same simulation |

Compare is the one to hold on during the pitch: left pane looks calm, right pane has 40
red buildings.

> A note on what *not* to claim: simply counting everything inside the failed towers'
> circles gives 723 buildings against a true 40, which looks like a far more impressive
> number and is a strawman — no operator reasons that way, because they can see the
> neighbouring circles overlapping. Both flat readings are wrong; only the one that
> leaves people off the air is worth putting on a slide. Locked down in
> `tests/test_coverage.py`.

---

## Things that turned out to be false

Recorded because each one would have been repeated on stage as fact.

**Height imputation is much weaker than it first appeared.** Three successive numbers
were wrong:

| reported MAE | why it was wrong |
|---|---|
| 33.4 m | neighbour features were computed from all labels, leaking the test set |
| 3.2 m | the wider training area *contains* the CBD — all 362 surveyed buildings appeared twice, so CV scored a building against its own duplicate at distance zero |
| 3.6 m | after dedup, still scored on the wider area, which is mostly identical terraced houses whose neighbours share their exact height |

**The defensible figure is MAE 36.0 m, median error 22 m, R² 0.23**, cross-validated on
held-out *CBD* buildings only. Adding 14,394 wider-area labels changed it by 0.3 m —
suburban heights do not transfer to high-rise. Production answer is LiDAR/DSM, not more
OSM. Imputed buildings are tinted differently in the UI so a guess never reads as a survey.

**MTTR measured from detection made early detection worthless by construction.** An
outage found after an hour and fixed in twenty minutes scored better than one caught
instantly and fixed in twenty-five. The clock now starts when service broke.

**Isolation Forest alone caused 18 false alarms per 2,000 samples.** `contamination=0.02`
means it flags ~2% of *normal* data by design. It is now calibrated against its own
training window and can only corroborate, never trip the alarm alone.

**An EWMA fast enough to track drift is fast enough to learn a fault as normal.** At
α=0.05 a 16 dB RSSI collapse was never detected — the mean slid down with it. α=0.002 was
chosen by sweeping both failure modes (see the table in `edge/detector.py`).

**Two of four crews could reach nothing.** Depots had snapped onto road stubs clipped by
the AOI boundary. Snapping is now restricted to the largest *strongly* connected component.

---

## Architecture

```
Browser — deck.gl (vendored)          buildings · outage shadow · towers · crew routes
    ▲ WebSocket, 4 Hz state deltas
FastAPI — simulation clock · coverage · dispatch · metrics
    ▲ events only, never raw telemetry
Edge agents — per-tower telemetry + local anomaly inference
```

The browser is a pure renderer; the server owns the clock. Building geometry goes over
HTTP once, so only mutable state rides the socket.

| path | role |
|---|---|
| `twinsync/geo.py` | numpy geometry, local projection, grid index |
| `twinsync/coverage.py` | 3D LOS raycasting, fingerprinted cache |
| `twinsync/routing.py` | street graph, A\* on travel time, congestion |
| `twinsync/dispatch.py` | assignment, batching, preemption (`smart=False` = baseline) |
| `twinsync/sim.py` | headless simulation, both arms |
| `edge/detector.py` | three detectors on two cadences |
| `edge/intelligence.py` | simulated ST-DBSCAN + LightGBM output contracts (v0.1) |
| `scripts/` | one-time data pipeline |

### Prototype infrastructure note

To keep judge setup friction low, this v0.1 prototype intentionally avoids a required
container/database stack. It uses lightweight GeoJSON artifacts (for world state) and
in-memory Python state (for runtime state) in place of mandatory PostGIS/TimescaleDB
services. Simulated MQTT-style edge payloads are passed directly into FastAPI handlers
rather than through an external broker in this repository.

### Two performance notes

`IsolationForest.score_samples` costs **2.2 ms per call**, and that is almost entirely
fixed overhead — scoring 15 rows costs the same as scoring one. At 10 Hz across 15 towers
it would need a third of a CPU core. Running the cheap tests every sample and the forest
on a duty cycle made the fleet **15× cheaper** (2211 µs → 150 µs) with no loss in
detection latency.

deck.gl reads its container's size when it builds its canvas. Constructing it before
layout settles yields a canvas that reports correct dimensions and **never paints** — the
3D scene is silently blank while the HUD, being plain DOM, looks perfectly healthy. `boot()`
now waits for `load` plus two animation frames.

---

## Reproducing the data

```bash
python scripts/fetch_osm.py --bbox 3.140,101.705,3.162,101.725 --out data
python scripts/impute_heights.py --in data/buildings.geojson \
       --train-extra data/train_buildings.geojson --report
python scripts/place_towers.py --n 15
python scripts/make_scenario.py
```

Overpass is unreliable — during this build it returned 504s, 429s and SSL errors, and one
tile needed five attempts across all four mirrors. The fetcher rotates mirrors, backs off,
tiles the AOI and caches every tile, so a re-run resumes. **The committed GeoJSON is the
artifact; the demo never touches the network.**
