# TwinSync — closing the gap between the pitch and the code

## Context

TwinSync is a 4-day hackathon prototype for the **ASEAN GeoAI Fusion** competition: a real-time
operational digital twin for telecom field efficiency over the KL CBD. The engineering that
exists is genuinely good — 3D line-of-sight raycasting against extruded buildings
([twinsync/coverage.py](twinsync/coverage.py)), a real networkx A\* road router
([twinsync/routing.py:223](twinsync/routing.py#L223)), a three-tier streaming anomaly detector
([edge/detector.py](edge/detector.py)), and an honest A/B harness that runs the identical
scenario through both dispatch arms.

The problem is that the four claims a GeoAI judge will probe hardest are the four that are
**not implemented**, and [README.md:34-64](README.md#L34-L64) currently admits it:

| Claim on the slide | What the code actually does |
|---|---|
| ST-DBSCAN fault localisation | `sha1(sorted(failed_towers) + time_bucket)` — [edge/intelligence.py:40](edge/intelligence.py#L40). No eps, no min_pts, no coordinates. `self.world` is stored and never read. |
| LightGBM 7-day risk scoring | A hardcoded 4-term weighted sum — [edge/intelligence.py:50](edge/intelligence.py#L50). `lightgbm` is never imported. |
| Quantized ONNX/TFLite edge inference | No model file of any kind exists in the repo. `onnxruntime` is never imported. |
| Copernicus DEM GLO-30 / Z-axis | No raster, no elevation. Ground is implicitly flat at z=0 ([coverage.py:101](twinsync/coverage.py#L101)). |
| SHAP explainability | A hardcoded HTML string — [web/app.js:357](web/app.js#L357). |
| "15× cheaper" / "2.2 ms per call" | No benchmark script exists, and the two docs disagree with each other ([detector.py:21](edge/detector.py#L21) says 20×, [README.md:195](README.md#L195) says 15×). |

There is also no `requirements.txt`, and the one install line in the README omits `numpy`,
`scikit-learn` and `scipy` (the demo cannot start without sklearn) while listing `shapely`,
which is imported nowhere.

**Intended outcome:** every architectural claim has a file, a benchmark, or a test behind it;
the geospatial story becomes real multi-source fusion (Copernicus DEM + OSM + monsoon weather);
the live demo gains an operator-triggerable incident cascade; and the results slide carries
telco-legible ROI numbers instead of only MTTR.

**Priority order (confirmed):** real models → GeoAI fusion → demo UX → business KPIs.
One exception to that ordering is called out in Phase 1.

---

## Constraints to preserve

These are deliberate design decisions in the existing code. Do not break them.

1. **The demo runs with zero network.** Vendored deck.gl, no map tiles, no CDN, no API keys
   ([README.md:23](README.md#L23), [server.py:138-142](twinsync/server.py#L138-L142)). All new
   data must be baked into the repo at build time and read from disk at runtime.
2. **Determinism.** Same seed → same run; the pitch cannot depend on chance
   ([make_scenario.py:2-6](scripts/make_scenario.py#L2-L6)).
3. **Honest labelling.** The README's scope-disclosure section is a strength, not a weakness.
   Models trained on synthetic data must stay labelled as such — we are moving claims from
   *"simulated"* to *"trained on simulated data, assumptions documented"*, not to *"production"*.
4. **Reuse what exists.** `LocalFrame` for lon/lat↔metres ([geo.py:35](twinsync/geo.py#L35)),
   `PolygonSet` grid index ([geo.py:136](twinsync/geo.py#L136)), `Route.distance_m`
   ([routing.py:47](twinsync/routing.py#L47)), `network.set_congestion`
   ([routing.py:165](twinsync/routing.py#L165)), `engine.batched_count`, the coverage cache
   fingerprint ([coverage.py:227](twinsync/coverage.py#L227)), and the existing
   `POST /api/fault/{tower_id}` + `POST /api/control/speed` endpoints, which are already
   implemented and simply never called by the frontend.

---

## Phase 0 — Foundations (do first, ~30 min)

- **`requirements.txt`** at repo root, split runtime vs build:
  - runtime: `fastapi`, `uvicorn[standard]`, `websockets`, `networkx`, `numpy`,
    `scikit-learn`, `onnxruntime`, `lightgbm`
  - build/train (`requirements-dev.txt`): `scipy`, `rasterio`, `skl2onnx`, `onnx`, `shap`,
    `matplotlib`, `pytest`, `requests`
  - Drop `shapely` (unused). Pin minor versions.
- **`models/`** directory, committed. Artifacts are small (<100 KB total) and shipping them
  is the entire point — a judge must be able to `ls models/`.
- Update the install block at [README.md:18](README.md#L18).

---

## Phase 1 — Real models

### 1a. Terrain prerequisite (out of priority order, deliberately)

`terrain_slope` and flood-prone road segments are inputs to the risk model, so the *data*
half of Phase 2 must land first. Only the fetch script runs here; the Fresnel/coverage
integration stays in Phase 2.

**New: `scripts/fetch_dem.py`**
- Pulls the Copernicus DEM GLO-30 COG for the KL tile from the public AWS open-data bucket
  (`copernicus-dem-30m`, no credentials): `Copernicus_DSM_COG_10_N03_00_E101_00_DEM`.
- Windowed-read the AOI with `rasterio`, resample to a regular grid over the world bbox.
- Writes **`data/terrain.json`**: `{bbox, nx, ny, cell_m, elevations: [...]}` — a ~60×60
  float grid at 30 m, ≈40 KB. Also stamps `ground_elev` onto every feature in
  `data/buildings.geojson` and `data/towers.geojson`, mirroring how
  [impute_heights.py:347](scripts/impute_heights.py#L347) writes `height` +
  `height_source` back into the GeoJSON.
- `--synthetic` flag generates a deterministic seeded surface as a fallback, stamping
  `dem_source: "synthetic"` so it can never be mistaken for real data.

**New: `twinsync/terrain.py`** — runtime side, numpy only, no rasterio:
- `Terrain.load(path)`, `elevation_at(x, y)` (bilinear), `profile(a_xy, b_xy, n)` sampled
  terrain profile between two points, `slope_at(x, y)` (degrees, from the gradient),
  `is_low_lying(x, y, percentile)` for flood proneness.
- Wire `ground_elev` into `Tower` / `Building` in [twinsync/world.py](twinsync/world.py); add
  `Tower.base_z = ground_elev` and `Tower.antenna_z = ground_elev + antenna_height`.

### 1b. Real ST-DBSCAN

**New: `twinsync/stdbscan.py`** (moved out of `edge/`, since it fuses fleet-wide state —
`edge/intelligence.py` shrinks to a thin adapter).

Implement Birant & Kut (2007) ST-DBSCAN over alarm points
`(x_m, y_m, z_m, t_s, alarm_type)`:
- `eps_spatial` (metres, default ~900 — tuned so the KL-03/KL-13 pair in the scenario
  co-clusters and KL-09 does not), `eps_temporal` (seconds, default 600), `min_pts` (2).
- 3D spatial distance uses **antenna phase-centre elevation** `ground_elev + antenna_height`,
  so the DEM genuinely enters the clustering — this is the "Z-axis" claim made real.
- Two alarms are ST-neighbours iff spatial ≤ `eps_spatial` **and** |Δt| ≤ `eps_temporal`
  **and** `alarm_type` is compatible (same profile, or in a documented cascade set such as
  `power_failure → backhaul_congestion`). Compatibility table lives in one module constant.
- Real DBSCAN core/border/noise semantics: singleton alarms return label `-1` (noise), which
  is the discriminating behaviour the current hash-stub cannot express.
- **Streaming wrapper `AlarmClusterer`**: rolling buffer with a retention window (default
  30 sim-minutes), re-clusters on each new alarm (fleet is 15 towers, cost is negligible),
  and preserves **label stability** across re-clusterings by Jaccard-matching new clusters to
  previous ones — otherwise cluster IDs flicker in the UI every frame.
- Returns `LocalisationResult(cluster_id, members, is_noise, span_m, span_s, model="st-dbscan-v1")`.

Call site: [sim.py:163](twinsync/sim.py#L163) swaps `simulate_st_dbscan` for the real
clusterer. Log line at [sim.py:183](twinsync/sim.py#L183) gains the cluster span so the
localisation is visible in the demo log.

**New: `tests/test_stdbscan.py`** — co-located simultaneous alarms cluster; spatially far
alarms do not; temporally far alarms do not; a lone alarm is noise; incompatible alarm types
stay separate; label stability holds when a member is added.

### 1c. Quantized ONNX edge inference + benchmark

**New: `scripts/train_edge_model.py`**
- Generates healthy telemetry across all 15 towers from
  [edge/telemetry.py](edge/telemetry.py), runs it through the detector's existing
  `_z_scores` ([detector.py:142](edge/detector.py#L142)) so the model input is the **6 EWMA
  z-scores** — tower-agnostic, needs no per-tower normalisation at inference, and reuses
  state the detector already computes for free.
- Trains a small autoencoder `6 → 16 → 8 → 16 → 6` with sklearn's `MLPRegressor` (X→X),
  avoiding a torch dependency entirely. Reconstruction MSE is the anomaly score; threshold
  = a high percentile of healthy reconstruction error, stored in the metadata sidecar.
- Exports via `skl2onnx` → `models/edge_anomaly_fp32.onnx`, then
  `onnxruntime.quantization.quantize_dynamic` (INT8) → **`models/edge_anomaly_int8.onnx`**
  (the shipped artifact, ~15 KB). Sidecar `models/edge_anomaly_meta.json` carries threshold,
  input order, training date, sample count, and a sha256 of the artifact.

**Modify [edge/detector.py](edge/detector.py)** — replace the duty-cycled IsolationForest
confirm stage, keeping the architecture narrative in the module docstring intact:
- cheap gate (every sample): EWMA z + Mahalanobis — **unchanged**
- confirm stage: `onnxruntime.InferenceSession` on the INT8 model, still duty-cycled by
  `FOREST_PERIOD` and still requiring live Mahalanobis corroboration
  ([detector.py:212](edge/detector.py#L212)) so a stale verdict cannot manufacture a state change
- fallback: if `onnxruntime` import fails, fall back to today's per-tower IsolationForest
  path so the demo can never be bricked by an install problem on the demo machine
- Rename `forest_*` fields on `Verdict` to neutral `confirm_*` names, retaining `forest_score`
  as a deprecated alias for one release so `sim.py` and tests keep working.

**New: `scripts/bench_edge.py`** — this is the artifact that backs the latency claims:
- `time.perf_counter_ns` over N warmup + N measured iterations per stage: EWMA z,
  Mahalanobis, ONNX INT8, ONNX FP32, sklearn IsolationForest. Reports **p50/p95/p99**, not
  just a mean.
- Memory: model file size on disk, `tracemalloc` peak, and RSS delta for a **15-tower fleet**
  of live detector instances — that is the number that supports the "<50 MB RAM" claim.
- Headroom: achieved samples/sec vs the 10 Hz × 15 towers the deployment needs.
- Writes `data/bench_edge.json` **and prints a markdown table for direct paste into the
  README**, stamped with CPU model, core count, Python and onnxruntime versions.
- This regenerates the hand-written numbers in
  [detector.py:16-22](edge/detector.py#L16-L22) and
  [README.md:195-199](README.md#L195-L199) and resolves the 20×-vs-15× contradiction.

> **TFLite:** ship ONNX only. Adding TensorFlow for a second export of the same 15 KB
> network is a large dependency for no additional evidence. Remove the TFLite mention from
> [README.md:47](README.md#L47) rather than half-shipping it.

**New: `tests/test_edge_onnx.py`** — artifact loads; INT8 and FP32 agree within tolerance;
healthy telemetry scores below threshold and each of the 4 fault profiles scores above;
the fallback path activates cleanly when onnxruntime is monkeypatched out. Also adds the
first-ever tests for `EdgeDetector` itself (250 lines, currently zero coverage).

### 1d. Trained LightGBM + real explainability

**New: `scripts/train_risk_model.py`** — physics-informed synthetic generator, ~50k
tower-week records. Every hazard term documented in the module header with its rationale, so
a judge can argue with the assumptions rather than be misled by them.

Features (the ones named in the brief, plus the ones the DEM and weather now unlock):
`asset_age_years`, `recent_thermal_cycles`, `peak_temperature_c`, `vswr_trend_30d`,
`terrain_slope_deg` *(DEM)*, `elevation_m` *(DEM)*, `rainfall_intensity_mm_hr`,
`lightning_density_per_km2`, `humidity_pct`, `days_since_maintenance`, `load_factor`,
`subscribers_served`, `encroachment_risk`.

Label: binary failure within a 7-day horizon, drawn from a documented hazard function —
Arrhenius-style thermal ageing, VSWR-trend degradation, monsoon rainfall/lightning exposure,
and a terrain term. Grouped train/test split **by tower** to prevent leakage (the same
leakage discipline as `HeightImputer.cross_validate` at
[impute_heights.py:194](scripts/impute_heights.py#L194)).

Artifacts, all committed:
- `models/risk_lgbm.txt` — native booster
- `models/risk_feature_importance.png` — gain-based importance
- `models/risk_shap_summary.png` — real SHAP beeswarm, replacing the hardcoded string at
  [web/app.js:357](web/app.js#L357)
- `models/risk_metrics.json` — ROC-AUC, PR-AUC, Brier score, calibration curve
- `models/risk_model_card.md`

**New: `twinsync/risk.py`** — `RiskScorer.load()` reads the booster once at startup,
`score(tower, weather, now)` assembles the feature vector and returns
`RiskResult(score, band, model="lightgbm-v1", top_factors=[...])` where `top_factors` are
**per-incident SHAP values** computed live via `booster.predict(pred_contrib=True)` (cheap,
no `shap` needed at runtime). This is what makes the tooltip explainability genuine.

Call site: [sim.py:164](twinsync/sim.py#L164). `ai_model_source` at
[sim.py:175](twinsync/sim.py#L175) changes from `"simulated-v0.1"` to
`"lightgbm-v1+st-dbscan-v1"`.

> Keep `twinsync/priority.py` as-is. It is the deterministic dispatch ordering and must stay
> auditable; the ML risk score is advisory context shown to the operator, not the dispatcher.
> Say this explicitly in the README — "the ML never silently drives the truck" is a strong
> answer to the human-in-the-loop question judges ask.

---

## Phase 2 — GeoAI fusion

### 2a. Fresnel-zone clearance over DEM terrain

**Modify [twinsync/coverage.py](twinsync/coverage.py):**
- Add `Tower.frequency_mhz` (access default 2100; backhaul 18000) in
  [world.py:72](twinsync/world.py#L72).
- First Fresnel radius `r = sqrt(λ·d1·d2 / (d1+d2))`, λ = c/f. Standard **60 % of F1 must be
  clear** criterion.
- Rewrite `_is_blocked` ([coverage.py:63](twinsync/coverage.py#L63)) to compute *clearance*
  rather than a hard hit: obstacle top = `ground_elev + height`; blocked when
  `ray_z − obstacle_top < 0.6·r` at the crossing. Terrain itself becomes an obstacle by
  sampling `Terrain.profile()` along the path.
- New `link_profile(a, b)` returning the terrain + building profile between two towers, for
  backhaul LOS and for the UI.
- Extend the cache fingerprint at [coverage.py:227](twinsync/coverage.py#L227) with terrain
  hash + frequency so a stale `data/coverage_cache.json` cannot silently survive.

> **Warning — this changes the headline number.** Fresnel clearance is stricter than a hard
> shadow, so the "3 buildings (2D) vs 40 buildings (3D)" figure at
> [README.md:96-104](README.md#L96-L104) will move, and the 9 tests in
> [tests/test_coverage.py](tests/test_coverage.py) that lock those values down will fail by
> design. Re-run, update the assertions, regenerate `data/results.json`, and update the
> README. Budget time for this — it is the most invasive change in the whole plan.

### 2b. Tropical weather & monsoon resilience

**Modify [scripts/make_scenario.py](scripts/make_scenario.py)** — add a `weather` block:
```json
"weather": {
  "profile": "northeast_monsoon",
  "storm_cells": [
    {"lon": 101.71, "lat": 3.148, "radius_m": 1800, "peak_mm_hr": 95,
     "start_s": 900, "duration_s": 1500,
     "drift_bearing_deg": 215, "drift_kmh": 18}
  ],
  "baseline": {"humidity_pct": 82, "wind_kmh": 11}
}
```
Moving storm cells (rather than a scalar) give spatially varying rain, which is what makes
this geospatial rather than a slider — and they animate beautifully on the map.

**New: `twinsync/weather.py`** — `WeatherField.at(x, y, t)` → `{rainfall_mm_hr,
lightning_per_km2_hr, humidity_pct, wind_kmh}` by superposing drifting Gaussian cells.

Four documented effects, each wired into a different subsystem — this is the fusion story:

1. **Backhaul rain fade** — ITU-R P.838 specific attenuation `γ = k·R^α` dB/km applied to
   the *microwave backhaul* link budget (18 GHz), degrading `throughput_mbps` in
   [edge/telemetry.py](edge/telemetry.py). Deliberately **not** applied to the 2.1 GHz access
   RSSI, where rain fade is negligible. State this in the code comment — being precise about
   where rain does and does not matter is more credible than a blanket multiplier, and
   backhaul-under-monsoon is the actual ASEAN operator pain point.
2. **Lightning** raises the stochastic `power_failure` hazard, and is a risk-model feature.
3. **Humidity + rain** drive `temperature_c` (cooling efficiency) and `vswr` (water ingress)
   drift.
4. **Flooding → dispatch.** Rain above a threshold over **low-lying road segments identified
   from the DEM** multiplies travel time — reusing `network.set_congestion`
   ([routing.py:165](twinsync/routing.py#L165)) with a new `low_lying=True` selector. DEM +
   rainfall + road graph → the router reroutes crews around flooded arterials. This single
   path touches three data sources and is the strongest "GeoAI Fusion" demo moment in the
   project; make sure the demo script narrates it.

**New: `tests/test_weather.py`, `tests/test_terrain.py`** — cell drift is deterministic under
seed; rain fade increases monotonically with rate; DEM bilinear interpolation matches known
grid corners; flood selector picks only low-lying segments.

---

## Phase 3 — Demo UX

All layers needed are already inside the vendored `web/vendor/deck.gl.min.js` — **no new
frontend dependencies, no build step.**

### 3a. Chaos / incident trigger panel
Add a control cluster to [web/index.html:28-40](web/index.html#L28-L40) and handlers near
[web/app.js:607](web/app.js#L607):
- **"Simulate Monsoon Storm"** → new `POST /api/storm` (spawns a storm cell over the AOI)
- **"Trigger Tower Outage"** → the existing, already-implemented
  `POST /api/fault/{tower_id}?profile=` ([server.py:170](twinsync/server.py#L170)) — a tower
  picker plus the 4 profiles from [telemetry.py:30-55](edge/telemetry.py#L30-L55)
- **Speed slider 1×–60×** → the existing `POST /api/control/speed`
  ([server.py:163](twinsync/server.py#L163)), currently unreachable from the UI
- Fix the pause/resume desync at [app.js:608](web/app.js#L608) (it infers paused state from
  its own button label, so a Reset leaves it stuck reading "Resume")

The narrated cascade this unlocks, end to end:
`storm cell drifts in → rain fade + lightning degrade towers → edge ONNX confirms anomalies
→ ST-DBSCAN groups them into one localised cluster instead of N isolated tickets → LightGBM
re-scores neighbouring towers → flooded low-lying roads reprice → dispatch reroutes and
preempts`.

### 3b. Visual showcase
- **`TripsLayer`** for crew trails — vehicles currently snap at 4 Hz with no interpolation
  and no `transitions` prop ([app.js:261-274](web/app.js#L261-L274)).
- **Coverage cones/volumes**: extruded `PolygonLayer` wedges per tower, driven by the real
  LOS result — surfaced through the already-implemented-but-never-called
  `GET /api/coverage/{tower_id}` ([server.py:185](twinsync/server.py#L185)), whose docstring
  literally promises an on-screen toggle that does not exist.
- **Storm overlay**: animated `ScatterplotLayer` / `HeatmapLayer` for cells, plus a
  rainfall-intensity ground layer and a flooded-road highlight on the existing roads
  `PathLayer`.
- **Terrain shading** on the ground plane from `data/terrain.json`, so the DEM is *visible*,
  not just computed.
- **Real SHAP** in the tower tooltip ([app.js:355-358](web/app.js#L355-L358)) — replace the
  hardcoded attribution string with the live `top_factors` from `twinsync/risk.py`.
- Fix the dead CSS rule at [web/style.css:273](web/style.css#L273): it targets `body.split`
  but `setViewMode()` sets `mode-split` ([app.js:325](web/app.js#L325)), so panels currently
  overlap the map panes in Compare mode — the exact thing that rule was written to prevent.

### 3c. Server additions
- `POST /api/storm` — inject a storm cell
- `GET /api/models` — **model card endpoint**: artifact names, versions, sha256 hashes,
  training dates, held-out metrics. A judge-facing "prove it" button.
- `GET /api/metrics` — live KPI rollup for the new dashboard tiles

---

## Phase 4 — Business impact & ROI

**Modify [twinsync/metrics.py](twinsync/metrics.py).** Existing fields stay; the MTTR
docstring at [metrics.py:45-55](twinsync/metrics.py#L45-L55) is well-reasoned and should not
change. Add:

| KPI | Derivation |
|---|---|
| **MTTD** | `detected_at − fault_started_at` — already collected as `detection_minutes` at [metrics.py:110](twinsync/metrics.py#L110), just needs surfacing under the right name |
| **MTTL** | cluster-assignment time − first alarm in that ST-DBSCAN cluster (new, from Phase 1b) |
| **MTTR p50 / p90** | percentiles alongside the existing mean — a mean over 4 incidents is thin on its own |
| **travel_distance_km** | sum of `Route.distance_m` per crew — already available on the `Route` dataclass ([routing.py:47](twinsync/routing.py#L47)), simply never accumulated |
| **fuel_litres** | `distance_km × 0.11` (urban light commercial van, incl. idling — constant documented at module level) |
| **co2_kg** | `fuel_litres × 2.68` (diesel, DEFRA/DESNZ factor — cited in the constant's comment) |
| **truck_roll_reduction_pct** | from the existing `truck_rolls` / `engine.batched_count` |
| **crew_utilisation_pct** | busy seconds / elapsed seconds, per crew and fleet-wide |
| **sla_uptime_pct** | `1 − subscriber_minutes_lost / (total_subscribers × duration_min)` |
| **annualised_savings** | extrapolation from the 1-hour window, with the cost-per-truck-roll assumption stated inline and in the README |

Extend `Comparison.render()` ([metrics.py:161](twinsync/metrics.py#L161)) with the new rows,
`as_dict()` for `data/results.json`, and add matching KPI tiles to the dashboard
(`renderKpis()`, [app.js:370](web/app.js#L370)).

**New: `tests/test_metrics.py`** — currently there is no test file for `metrics.py`,
`dispatch.py`, `sim.py`, `priority.py` or `world.py`. Cover the derived KPIs against a
hand-computed fixture at minimum.

---

## Documentation (not optional)

- Rewrite the scope-disclosure section at [README.md:34-64](README.md#L34-L64): move
  ST-DBSCAN, ONNX and LightGBM out of "simulated" into "**trained on simulated data,
  assumptions documented in the training script**". Sentinel-2 NDVI stays labelled simulated
  ([sim.py:96](twinsync/sim.py#L96)) — do not quietly upgrade it.
- **New: `MODEL_CARDS.md`** — one card per artifact: inputs, training data provenance,
  held-out metrics, known limitations, and an explicit "trained on synthetic data" banner.
- Paste the real `scripts/bench_edge.py` table into the README, replacing the two
  contradicting hand-written figures.
- Update the architecture diagram at [README.md:163-172](README.md#L163-L172) to show the DEM
  and weather inputs, and fix "state deltas" → "full snapshots" (`sim.snapshot()` sends a
  complete frame every tick, [sim.py:317](twinsync/sim.py#L317)).

---

## Verification

Run in this order; each step gates the next.

```bash
# 0. deps
pip install -r requirements.txt -r requirements-dev.txt

# 1. data pipeline (once; --synthetic if the machine is offline)
python scripts/fetch_dem.py --data data
python scripts/make_scenario.py --out data/scenario.json     # now includes weather

# 2. train + export, artifacts land in models/
python scripts/train_edge_model.py --out models/
python scripts/train_risk_model.py --out models/
ls -la models/            # must show .onnx, .txt, .png, .json, model card

# 3. the evidence
python scripts/bench_edge.py --iterations 5000 --fleet 15
#    -> assert INT8 p95 < 15 ms and 15-tower fleet RSS < 50 MB; paste table into README

# 4. full suite (test_coverage assertions WILL need updating after Fresnel — expected)
python -m pytest -q

# 5. headless A/B, regenerate the results slide
rm -f data/coverage_cache.json     # force recompute with terrain + Fresnel
python -m twinsync.sim --scenario data/scenario.json --seed 42 --json data/results.json

# 6. live demo
python -m uvicorn twinsync.server:app --port 8000
```

**Manual checks in the browser at `localhost:8000`:**
1. `GET /api/models` returns real artifact hashes and metrics.
2. Terrain shading is visible; buildings sit at their DEM ground elevation, not z=0.
3. Click **Simulate Monsoon Storm** → storm cell drifts across the map, rainfall overlay
   animates, low-lying roads highlight as flooded, and crew routes visibly change.
4. Click **Trigger Tower Outage** on two nearby towers within the temporal window → the
   incident panel shows **one** ST-DBSCAN cluster ID with 2 members, not two isolated
   incidents. Trigger a third far-away tower → it comes back as noise (`-1`), a distinct
   incident. *This is the single clearest proof the clustering is real.*
5. Hover a degraded tower → SHAP factors are numeric and vary per tower (the old hardcoded
   string was identical everywhere — an easy tell).
6. Crew vehicles glide along road polylines with `TripsLayer` trails instead of snapping.
7. Compare (`3`) mode: side panels no longer overlap the map panes.
8. Speed slider moves the clock; pause/resume label stays correct across a Reset.

**Regression guard:** `data/results.json` before and after should show the same *direction*
on every KPI. If Fresnel + weather flip a metric so TwinSync looks worse than baseline, that
is a real finding worth reporting honestly — not something to tune away.

---

## Risks

| Risk | Mitigation |
|---|---|
| Fresnel + terrain change the headline 2D-vs-3D number, breaking 9 tests and the README | Expected and planned for in 2a. Re-run, update assertions, update the README with the new number. Do not fudge the assertions to preserve the old figure. |
| `rasterio` install pain on Windows | Build-time only. `scripts/fetch_dem.py --synthetic` and the committed `data/terrain.json` mean the demo machine never needs it. |
| `onnxruntime` unavailable at demo time | Explicit fallback to the current IsolationForest path in `detector.py`, covered by a test. |
| Synthetic training data reads as fake to a judge | Front-foot it: document every hazard assumption in the script header, ship model cards, and say plainly that no public telco failure dataset exists. Stated assumptions beat unstated ones. |
| Scope is large for the time available | Phases are independently shippable in the stated priority order. Phase 1 alone closes the biggest credibility gap; Phase 4 is the cheapest and can be pulled forward if the results slide needs numbers sooner. |
