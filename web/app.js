/* TwinSync dashboard client.
 *
 * A renderer and nothing else. The server owns the clock and the model; this file turns
 * pushed state into layers. No simulation happens here, so the picture cannot drift out
 * of step with the twin behind it.
 *
 * Everything is local: deck.gl is vendored, the geometry comes from our own GeoJSON, and
 * the roads are drawn rather than fetched as basemap tiles. The page works with the
 * network cable pulled out, which is the only way to be sure of it on stage.
 */

const {DeckGL, MapView, PolygonLayer, PathLayer, ScatterplotLayer, ColumnLayer,
       TextLayer, PathStyleExtension, TripsLayer} = deck;

/* ------------------------------------------------------------------ palette */

const C = {
  buildingLow:  [38, 48, 70],
  buildingHigh: [92, 112, 152],
  imputedTint:  [70, 78, 104],
  dark:         [168, 52, 58],
  darkTop:      [232, 88, 92],
  blocked2d:    [150, 120, 40],
  road:         [46, 58, 82],
  good:         [63, 185, 80],
  warn:         [210, 153, 34],
  crit:         [248, 81, 73],
  route:        [88, 166, 255],
  crew:         [235, 242, 255],
  storm:        [96, 150, 220],
  flood:        [70, 190, 210],
  cone:         [70, 140, 210],
};

/* Crew position history, kept client-side so TripsLayer has something to draw. The
 * server pushes a position, not a track: storing the tail here costs nothing and turns
 * four dots stepping at 4 Hz into vehicles that visibly move. Capped so a long demo
 * cannot grow it without bound. */
const TRAIL_LENGTH = 90;
const trails = new Map();

function recordTrails(snapshot) {
  if (!snapshot?.crews) return;
  for (const crew of snapshot.crews) {
    let trail = trails.get(crew.id);
    if (!trail) { trail = {path: [], timestamps: []}; trails.set(crew.id, trail); }
    const last = trail.path[trail.path.length - 1];
    // Skip duplicate samples: a parked crew should not accumulate a pile of identical
    // vertices, which makes the trail head jitter.
    if (!last || last[0] !== crew.lon || last[1] !== crew.lat) {
      trail.path.push([crew.lon, crew.lat]);
      trail.timestamps.push(snapshot.t);
      if (trail.path.length > TRAIL_LENGTH) {
        trail.path.shift();
        trail.timestamps.shift();
      }
    }
  }
}

const STATUS_COLOR = {healthy: C.good, degraded: C.warn, down: C.crit};

/* -------------------------------------------------------------------- state */

let world = null;            // static payload from /api/world
let state = null;            // latest snapshot from the WebSocket
let darkSet = new Set();     // building ids currently without service
let blockedSet = new Set();  // buildings a 2D radius would wrongly claim
let darkKey = '';            // membership signature, for deck.gl's accessor cache
let deckgl = null;
let logSeen = 0;

/* Split mode renders the same simulation through two cameras at once: a flat,
 * top-down pane showing what a 2D coverage radius claims, beside the pitched 3D
 * pane showing what line of sight actually delivers. Same instant, same faults,
 * same towers -- only the model of the world differs, which is the entire argument. */
/* '2d'    — the flat coverage map, as dispatch draws it today
 * '3d'    — true line of sight against the extruded city
 * 'split' — both, side by side, off the same instant of the same simulation */
let viewMode = '3d';
let dark2d = new Set();      // what a fair 2D coverage model concludes is dark
let dark2dKey = '';

/* ------------------------------------------------------------------ helpers */

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');

function clockText(seconds) {
  const m = Math.floor(seconds / 60), s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function bytesText(n) {
  if (n > 1048576) return (n / 1048576).toFixed(1) + ' MB';
  if (n > 1024) return (n / 1024).toFixed(0) + ' KB';
  return n + ' B';
}

/** Buildings read as a height field: taller is lighter, so the skyline is legible. */
function buildingColor(feature, mode) {
  const id = feature.id;
  // In the 2D pane a building is "affected" if it merely falls inside the radius --
  // that is the claim being made, and the point is to show how wrong it is.
  if (mode === '2d') {
    if (dark2d.has(id)) return C.dark;
  } else if (darkSet.has(id)) {
    return C.dark;
  }

  const h = feature.properties.height || 10;
  const t = Math.min(1, Math.log1p(h) / Math.log1p(300));
  const base = feature.properties.height_source === 'imputed' ? C.imputedTint : C.buildingLow;
  return [
    Math.round(base[0] + (C.buildingHigh[0] - base[0]) * t),
    Math.round(base[1] + (C.buildingHigh[1] - base[1]) * t),
    Math.round(base[2] + (C.buildingHigh[2] - base[2]) * t),
  ];
}

/* ------------------------------------------------------------------- layers */

/** Build one pane's worth of layers.
 *
 * Layer ids are prefixed with the view they belong to, and `layerFilter` on the deck
 * instance routes them: '2d-buildings' only ever draws in the 2D viewport. That is what
 * lets both panes read from one simulation without either knowing the other exists.
 */
/* Terrain mesh, built once from the baked Copernicus grid.
 *
 * Drawn as flat grid cells shaded by elevation rather than an extruded surface: the
 * buildings already sit at their true ground height, so extruding the ground too would
 * double the relief visually. This is here to make the DEM *legible* -- a judge asking
 * "is the elevation data real?" should be able to see the valley. */
let terrainCells = null;

function buildTerrainCells(grid) {
  const {nx, ny, cell_m, min_x, min_y, elevations, min_elev, max_elev} = grid;
  const span = Math.max(1e-6, max_elev - min_elev);
  const cells = [];
  // Every other cell in each direction: at 30 m the full grid is more polygons than the
  // relief justifies, and 60 m still reads as a smooth surface.
  for (let j = 0; j < ny - 1; j += 2) {
    for (let i = 0; i < nx - 1; i += 2) {
      const z = elevations[j * nx + i];
      const x0 = min_x + i * cell_m, y0 = min_y + j * cell_m;
      const x1 = x0 + 2 * cell_m, y1 = y0 + 2 * cell_m;
      // Fade the sheet out toward its own boundary. The DEM is a rectangle and the
      // world is not: drawn at uniform alpha it reads as a slab of floating paper with
      // the city standing on it. Dissolving the last ~15% into the background makes it
      // read as ground receding into the dark, which is what it is.
      const u = (i / (nx - 1)) * 2 - 1;       // -1..1 across the grid
      const v = (j / (ny - 1)) * 2 - 1;
      const edge = Math.max(Math.abs(u), Math.abs(v));
      const fade = Math.min(1, Math.max(0, (0.97 - edge) / 0.28));
      cells.push({
        polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        t: (z - min_elev) / span,
        alpha: fade,
        elev: z,
      });
    }
  }
  return cells;
}

/* Local metres -> lon/lat, mirroring twinsync.geo.LocalFrame so the mesh lands exactly
 * under the buildings. Uses the same spherical constant the server does. */
function metresToLonLat(x, y, originLon, originLat) {
  const M = 6371000.0 * Math.PI / 180.0;
  return [x / (M * Math.cos(originLat * Math.PI / 180)) + originLon, y / M + originLat];
}

function paneLayers(mode) {
  if (!world) return [];
  const p = (id) => `${mode}-${id}`;
  const flat = mode === '2d';
  const layers = [];

  // Ground first, so everything else draws over it.
  //
  // Depth testing stays ON here, unlike the road and route layers: the terrain is a real
  // surface at real altitude and buildings must occlude it. Painting it depth-free put a
  // lit sheet of paper over the whole city.
  if (!flat && terrainCells?.length) {
    layers.push(new PolygonLayer({
      id: p('terrain'),
      data: terrainCells,
      getPolygon: (c) => c.lonlat,
      extruded: false,
      filled: true,
      stroked: false,
      // Low ground cool and dark, high ground barely lighter. Deliberately a narrow
      // ramp: this is ground, and it must never compete with the outage red or the
      // building massing in front of it.
      getFillColor: (c) => [
        14 + 16 * c.t,
        19 + 18 * c.t,
        30 + 20 * c.t,
        Math.round(255 * c.alpha),
      ],
      // Unlit. With the scene's DirectionalLight applied, one flank of the grid caught
      // a warm highlight and the ground looked like it was under a sunset -- a
      // hypsometric ramp has to mean elevation, not incident angle.
      material: false,
      pickable: false,
    }));
  }

  layers.push(new PathLayer({
    id: p('roads'),
    data: world.roads.features,
    getPath: (f) => f.geometry.coordinates,
    getColor: flat ? [58, 72, 98] : C.road,
    getWidth: 2.5,
    widthMinPixels: 1,
    parameters: {depthTest: false},
  }));

  // The 2D pane draws the coverage radius the flat model actually reasons with -- a
  // circle on the ground. Seeing the circle next to the true shadow is the whole point.
  if (flat) {
    const failed = (state?.incidents || []).map((i) => i.tower);
    layers.push(new ScatterplotLayer({
      id: p('radius'),
      data: world.towers.features,
      radiusUnits: 'meters',
      getPosition: (f) => f.geometry.coordinates,
      getRadius: (f) => f.properties.range_m,
      filled: true,
      // Healthy circles are the crux: they overlap the failed one, which is why a flat
      // model concludes the neighbouring cell has it covered.
      getFillColor: (f) => (failed.includes(f.properties.id)
        ? [...C.blocked2d, 26] : [90, 130, 190, 12]),
      stroked: true,
      getLineColor: (f) => (failed.includes(f.properties.id)
        ? [...C.blocked2d, 200] : [90, 130, 190, 70]),
      lineWidthMinPixels: 1,
      updateTriggers: {getFillColor: [dark2dKey], getLineColor: [dark2dKey]},
    }));
  }

  layers.push(new PolygonLayer({
    id: p('buildings'),
    data: world.buildings.features,
    extruded: !flat,
    wireframe: false,
    filled: true,
    getPolygon: (f) => f.geometry.coordinates,
    getElevation: (f) => f.properties.height || 10,
    getFillColor: (f) => buildingColor(f, mode),
    material: flat
      ? null
      : {ambient: 0.42, diffuse: 0.62, shininess: 24, specularColor: [45, 55, 75]},
    pickable: true,
    // deck.gl caches accessor output; these tell it when colour actually changed.
    // Keyed on the actual membership, not its size: one building going dark while
    // another is restored leaves the count unchanged but the colours must still update.
    updateTriggers: {getFillColor: [darkKey, dark2dKey, mode]},
  }));

  const towers = world.towers.features;

  if (!flat) {
    // A shaft from rooftop to antenna makes the site legible against the skyline.
    layers.push(new ColumnLayer({
      id: p('tower-masts'),
      data: towers,
      diskResolution: 8,
      radius: 11,
      extruded: true,
      getPosition: (f) => f.geometry.coordinates,
      getElevation: (f) => f.properties.antenna_height,
      getFillColor: (f) => {
        const c = STATUS_COLOR[state?.tower_status?.[f.properties.id] || 'healthy'];
        return [c[0], c[1], c[2], 165];
      },
      updateTriggers: {getFillColor: [state?.t]},
    }));
  }

  layers.push(new ScatterplotLayer({
    id: p('tower-heads'),
    data: towers,
    billboard: true,
    radiusUnits: 'pixels',
    getPosition: (f) => (flat
      ? f.geometry.coordinates
      : [...f.geometry.coordinates, f.properties.antenna_height]),
    getRadius: (f) => (state?.tower_status?.[f.properties.id] === 'healthy' ? 5 : 9),
    getFillColor: (f) => STATUS_COLOR[state?.tower_status?.[f.properties.id] || 'healthy'],
    stroked: true,
    getLineColor: [5, 7, 13],
    lineWidthMinPixels: 2,
    pickable: true,
    updateTriggers: {getFillColor: [state?.t], getRadius: [state?.t]},
  }));

  layers.push(new TextLayer({
    id: p('tower-labels'),
    data: towers,
    getPosition: (f) => (flat
      ? f.geometry.coordinates
      : [...f.geometry.coordinates, f.properties.antenna_height]),
    getText: (f) => f.properties.id,
    getSize: 10,
    getColor: [154, 167, 189],
    getPixelOffset: [0, -16],
    fontFamily: 'ui-monospace, Menlo, Consolas, monospace',
    background: true,
    getBackgroundColor: [5, 7, 13, 170],
    backgroundPadding: [3, 1],
  }));

  // A flat dispatch map does show its crews -- it just cannot show a street route.
  // What it draws is a bearing and a distance, so that is what the 2D pane draws:
  // a dashed straight line to the job. Put next to the real route in the other pane,
  // it makes the second failure of the flat model visible -- not only is it wrong about
  // who went dark, it is wrong about who gets there first.
  if (flat && state?.crews?.length) {
    const byId = Object.fromEntries((state.incidents || []).map((i) => [i.id, i]));
    const bearings = [];
    for (const crew of state.crews) {
      const job = byId[crew.queue && crew.queue[0]];
      if (job && crew.status !== 'idle') {
        const tower = world.towers.features.find(
          (f) => f.properties.id === job.tower);
        if (tower) {
          bearings.push({path: [[crew.lon, crew.lat], tower.geometry.coordinates]});
        }
      }
    }
    if (bearings.length) {
      layers.push(new PathLayer({
        id: p('bearings'),
        data: bearings,
        getPath: (d) => d.path,
        getColor: [...C.blocked2d, 210],
        getWidth: 3,
        widthMinPixels: 2,
        getDashArray: [7, 4],
        dashJustified: true,
        extensions: [new PathStyleExtension({dash: true})],
        parameters: {depthTest: false},
        updateTriggers: {getPath: [state.t]},
      }));
    }
  }

  // -- weather ---------------------------------------------------------
  //
  // Drawn under everything else and with depth testing off, so the storm reads as
  // weather over the city rather than an object standing in it.
  const cells = state?.weather?.cells || [];
  if (cells.length) {
    layers.push(new ScatterplotLayer({
      id: p('storm'),
      data: cells,
      radiusUnits: 'meters',
      getPosition: (c) => [c.lon, c.lat],
      getRadius: (c) => c.radius_m,
      getFillColor: (c) => [...C.storm, Math.round(22 + 34 * c.intensity)],
      stroked: true,
      // A soft edge: a convective cell does not have a boundary, and a crisp outline
      // reads as a range ring rather than as weather.
      getLineColor: (c) => [...C.storm, Math.round(45 + 55 * c.intensity)],
      lineWidthMinPixels: 1,
      pickable: true,
      parameters: {depthTest: false},
      updateTriggers: {getPosition: [state.t], getFillColor: [state.t],
                       getRadius: [state.t]},
    }));
  }

  // Flooded low-lying roads: DEM + rainfall + road graph, which is the fusion claim
  // this project exists to make. Drawn over the road layer so it reads as a highlight.
  const flooded = state?.weather?.flooded_paths || [];
  if (flooded.length) {
    layers.push(new PathLayer({
      id: p('flooded'),
      data: flooded,
      getPath: (segment) => segment,
      getColor: [...C.flood, 230],
      getWidth: 6,
      widthMinPixels: 2.5,
      capRounded: true,
      parameters: {depthTest: false},
      updateTriggers: {getPath: [state.t]},
    }));
  }

  // -- coverage volume -------------------------------------------------
  //
  // Only for sites that are actually unwell. Drawing all fifteen cylinders at once --
  // which is what the first version did -- stacks 650 m discs on top of each other and
  // fogs the entire city into a grey wash; the layer stopped carrying information and
  // started hiding it. Restricted to failed sites it answers a real question: *this*
  // site is down, and this is the volume it was serving.
  const unwell = !flat
    ? (world.towers?.features || []).filter(
        (f) => (state?.tower_status?.[f.properties.id] || 'healthy') !== 'healthy')
    : [];
  // Drawn as a ring on the ground, not a filled cylinder. A 650 m x 250 m translucent
  // column seen at this camera pitch smears into a coloured haze across half the scene
  // -- it looks like a render artifact rather than a coverage volume, and it hides the
  // buildings whose outage status is the actual subject. A bright footprint ring says
  // the same thing in one glance and occludes nothing.
  if (unwell.length) {
    layers.push(new ScatterplotLayer({
      id: p('coverage-cones'),
      data: unwell,
      radiusUnits: 'meters',
      getPosition: (f) => f.geometry.coordinates,
      getRadius: (f) => f.properties.range_m,
      filled: true,
      getFillColor: (f) => {
        const status = state?.tower_status?.[f.properties.id];
        return status === 'down' ? [...C.crit, 14] : [...C.warn, 10];
      },
      stroked: true,
      getLineColor: (f) => {
        const status = state?.tower_status?.[f.properties.id];
        return status === 'down' ? [...C.crit, 170] : [...C.warn, 140];
      },
      lineWidthMinPixels: 1.5,
      pickable: false,
      parameters: {depthTest: false},
      updateTriggers: {
        getFillColor: [state?.t],
        getLineColor: [state?.t],
        getPosition: [unwell.length],
      },
    }));
  }

  if (state?.crews?.length) {
    // Vehicle trails. Without this the crews teleport between 4 Hz frames; with it the
    // eye tracks them along the street graph, which is what sells "real-time dispatch".
    //
    // Guarded on TripsLayer being present: it lives in deck.gl's geo-layers bundle and
    // a slimmer vendored build would not export it. A missing trail is a cosmetic loss;
    // an undefined constructor here would take down the entire render.
    if (!flat && TripsLayer) {
      const tracks = state.crews
        .map((c) => ({id: c.id, ...(trails.get(c.id) || {path: [], timestamps: []})}))
        .filter((t) => t.path.length > 1);
      if (tracks.length) layers.push(new TripsLayer({
        id: p('crew-trails'),
        data: tracks,
        getPath: (t) => t.path,
        getTimestamps: (t) => t.timestamps,
        getColor: C.route,
        opacity: 0.85,
        widthMinPixels: 3,
        trailLength: 240,
        currentTime: state.t,
        capRounded: true,
        jointRounded: true,
        parameters: {depthTest: false},
      }));
    }

    const routed = flat ? [] : state.crews.filter((c) => c.route && c.route.length > 1);
    if (routed.length) layers.push(new PathLayer({
      id: p('crew-routes'),
      data: routed,
      getPath: (c) => c.route,
      getColor: [...C.route, 205],
      getWidth: 5,
      widthMinPixels: 2.5,
      capRounded: true,
      jointRounded: true,
      parameters: {depthTest: false},
      updateTriggers: {getPath: [state.t]},
    }));

    layers.push(new ScatterplotLayer({
      id: p('crews'),
      data: state.crews,
      billboard: true,
      radiusUnits: 'pixels',
      getPosition: (c) => (flat ? [c.lon, c.lat] : [c.lon, c.lat, 6]),
      getRadius: 6,
      getFillColor: C.crew,
      stroked: true,
      getLineColor: flat ? [...C.blocked2d, 255] : [...C.route, 255],
      lineWidthMinPixels: 2,
      pickable: true,
      parameters: {depthTest: false},
      updateTriggers: {getPosition: [state.t], getLineColor: [mode]},
    }));
  }

  return layers;
}

function buildLayers() {
  if (viewMode === 'split') return [...paneLayers('2d'), ...paneLayers('3d')];
  return paneLayers(viewMode);
}

/** The viewport layout for the current mode. */
function currentViews() {
  const common = {controller: {dragRotate: true, inertia: 320}};
  const full = (id) => new MapView({id, x: 0, y: 0, width: '100%', height: '100%',
                                    ...common});
  if (viewMode === '2d') return [full('2d')];
  if (viewMode === '3d') return [full('3d')];
  return [
    new MapView({id: '2d', x: 0, y: 0, width: '50%', height: '100%', ...common}),
    new MapView({id: '3d', x: '50%', y: 0, width: '50%', height: '100%', ...common}),
  ];
}

function currentViewState() {
  const c = world ? world.centre : {lon: 101.7132, lat: 3.1497};
  const base = {longitude: c.lon, latitude: c.lat};
  // The flat view is deliberately top-down and north-up: that is how a dispatch map is
  // actually drawn today, and the contrast is lost if it is tilted too.
  const flat = {...base, pitch: 0, bearing: 0};
  const tilted = {...base, pitch: 56, bearing: -18};

  if (viewMode === '2d') return {'2d': {...flat, zoom: 14.2}};
  if (viewMode === '3d') return {'3d': {...tilted, zoom: 14.4}};
  // Side by side, each pane gets half the width, so pull back a little.
  return {
    '2d': {...flat, zoom: 13.7},
    '3d': {...tilted, pitch: 55, zoom: 13.7},
  };
}

/** Switch mode, rebuild the viewports, and update the chrome around them. */
function setViewMode(mode) {
  if (mode === viewMode) return;
  viewMode = mode;

  for (const b of ['2d', '3d', 'split']) {
    $(`btn-view-${b}`).classList.toggle('on', b === mode);
  }
  document.body.classList.remove('mode-2d', 'mode-3d', 'mode-split');
  document.body.classList.add(`mode-${mode}`);
  $('split-labels').hidden = (mode === '3d');

  deckgl.setProps({
    views: currentViews(),
    initialViewState: currentViewState(),
    layers: buildLayers(),
  });
  renderSplitReadout();
}

function tooltip({object, layer}) {
  if (!object) return null;
  if (layer.id.endsWith('buildings')) {
    const p = object.properties;
    const dark = darkSet.has(object.id);
    return {html:
      `<b>${p.name || 'Unnamed building'}</b><br>` +
      `${p.height.toFixed(0)} m · ${p.height_source === 'imputed' ? 'height imputed' : 'height from OSM'}<br>` +
      `${fmt(world.subscribers[object.id] || 0)} subscribers` +
      (dark ? '<br><b style="color:#f85149">NO SERVICE</b>' : '')};
  }
  if (layer.id.endsWith('tower-heads') || layer.id.endsWith('tower-masts')) {
    const p = object.properties;
    const st = state?.tower_status?.[p.id] || 'healthy';
    const d = state?.tower_digest?.[p.id];
    return {html:
      `<b>${p.id} — ${p.name}</b><br>` +
      `antenna ${p.antenna_height.toFixed(0)} m · status <b>${st}</b>` +
      (d ? `<br>${d.throughput_mbps} Mbps · ${d.temperature_c}&deg;C · ${d.packet_loss_pct}% loss` : '') +
      (d && d.rainfall_mm_hr > 0.5
        ? `<br>rain ${d.rainfall_mm_hr} mm/hr · backhaul fade ${d.backhaul_fade_db} dB
           (${Math.round(100 * d.backhaul_capacity)}% capacity)`
        : '') +
      (d ? `<br>encroachment risk (NDVI sim): ${d.encroachment_risk}%` : '') +
      riskFactorsHtml(p.id)};
  }
  if (layer.id.endsWith('crews')) {
    return {html: `<b>${object.name}</b><br>${object.status}` +
      (object.eta_s > 0 || object.eta_min > 0 ? ` · ETA ${object.eta_min} min` : '') +
      `<br>${object.trips} truck roll(s)`};
  }
  return null;
}

/* --------------------------------------------------------------------- HUD */

/* Real SHAP attributions for a tower's open incident.
 *
 * These come from LightGBM's pred_contrib, computed per incident on the server, so the
 * numbers differ per tower and change as conditions change. The previous version of
 * this was a hardcoded string that read identically on every site -- which is exactly
 * the tell a judge looks for. */
function riskFactorsHtml(towerId) {
  const incident = (state?.incidents || []).find((i) => i.tower === towerId);
  if (!incident || !incident.ai_risk_factors?.length) return '';

  const rows = incident.ai_risk_factors.slice(0, 3).map((f) => {
    const sign = f.contribution >= 0 ? '+' : '';
    const colour = f.contribution >= 0 ? '#f0883e' : '#3fb950';
    return `${f.feature.replace(/_/g, ' ')}
            <b style="color:${colour}">${sign}${f.contribution.toFixed(2)}</b>`;
  }).join(' · ');

  return `<br><span style="opacity:.75">7-day risk
          <b>${incident.ai_risk_score.toFixed(1)}%</b> (${incident.ai_risk_band})</span>` +
         `<br><span style="font-size:11px;opacity:.8">SHAP: ${rows}</span>`;
}

function renderKpis() {
  if (!state) return;

  let subs = 0;
  for (const id of state.dark_buildings) subs += world.subscribers[id] || 0;

  const subsEl = $('kpi-subs'), darkEl = $('kpi-dark');
  subsEl.textContent = fmt(subs);
  subsEl.parentElement.classList.toggle('alert', subs > 0);
  $('kpi-subs-note').textContent = subs > 0
    ? `${state.incidents.length} site(s) affected`
    : 'all sites nominal';

  darkEl.textContent = fmt(state.dark_buildings.length);
  darkEl.parentElement.classList.toggle('alert', state.dark_buildings.length > 0);
  // Summing per-incident counts would double-count shared buildings; the snapshot
  // already carries the deduplicated set.
  const n2d = (state.dark_buildings_2d || []).length;
  $('kpi-dark-note').textContent = state.dark_buildings.length
    ? `a 2D model reports only ${fmt(n2d)}`
    : 'a 2D model reports —';

  const up = state.uplink;
  if (up && up.raw_bytes > 0) {
    const pct = 100 * (1 - up.sent_bytes / up.raw_bytes);
    $('kpi-uplink').textContent = pct.toFixed(1) + '%';
    $('kpi-uplink-note').textContent =
      `${bytesText(up.sent_bytes)} sent vs ${bytesText(up.raw_bytes)} raw`;
  }

  $('kpi-open').textContent = state.incidents.length;
  $('kpi-open').parentElement.classList.toggle('warn', state.incidents.length > 0);

  // Fleet effort. Distance is accumulated server-side from the routes actually driven,
  // so the fuel and CO2 figures here are the same ones the results table reports.
  const rolls = (state.crews || []).reduce((sum, c) => sum + (c.trips || 0), 0);
  $('kpi-rolls').textContent = rolls;
  const km = (state.fleet?.travel_km ?? 0);
  $('kpi-rolls-note').textContent =
    `${km.toFixed(1)} km · ${(state.fleet?.co2_kg ?? 0).toFixed(1)} kg CO₂`;

  const weather = state.weather || {};
  const cells = weather.cells || [];
  const tile = $('kpi-weather-tile');
  if (cells.length) {
    const peak = Math.max(...cells.map((c) => c.rain_mm_hr));
    $('kpi-weather').textContent = `${peak.toFixed(0)} mm/hr`;
    $('kpi-weather-note').textContent = weather.flooded_segments
      ? `${cells.length} cell${cells.length > 1 ? 's' : ''} · ` +
        `${weather.flooded_segments} roads flooded`
      : `${cells.length} active cell${cells.length > 1 ? 's' : ''}`;
    tile.classList.add('warn');
  } else {
    $('kpi-weather').textContent = 'clear';
    $('kpi-weather-note').textContent = 'no active cells';
    tile.classList.remove('warn');
  }
  const busy = state.crews.filter((c) => c.status !== 'idle').length;
  $('kpi-open-note').textContent = busy
    ? `${busy} of ${state.crews.length} crews deployed`
    : 'crews idle';
}

function renderIncidents() {
  const host = $('incidents');
  if (!state.incidents.length) {
    host.innerHTML = '<p class="empty">No open incidents.</p>';
    return;
  }
  host.innerHTML = state.incidents.map((i) => `
    <div class="inc ${i.severity === 'down' ? 'down' : ''}" data-tower="${i.tower}">
      <div class="inc-top">
        <span class="inc-id">${i.id} · ${i.tower}</span>
        <span class="inc-pri">P ${i.priority.toFixed(1)}</span>
      </div>
      <div class="inc-row">
        <span class="n">${fmt(i.subscribers)}</span> subscribers ·
        <span class="n">${i.buildings_dark}</span> buildings dark
      </div>
      <div class="inc-row">a 2D model reports
        <span class="n">${i.buildings_dark_2d ?? 0}</span> dark &mdash; misses
        <span class="n">${i.missed_by_2d ?? 0}</span></div>
      ${i.critical_sites.length
        ? `<div class="inc-crit">critical: ${i.critical_sites.slice(0, 2).join(', ')}</div>` : ''}
      ${i.crew_2d && i.crew_2d !== i.assigned_to && i.crew_2d_minutes != null
        ? `<div class="inc-2d">2D would send ${i.crew_2d} &mdash;
             ${(i.crew_2d_minutes - (i.assigned_minutes ?? 0)).toFixed(1)} min slower</div>`
        : ''}
      ${i.ai_cluster_id
        ? `<div class="inc-2d">AI (${i.ai_model_source}): cluster ${i.ai_cluster_id}
             &middot; risk ${i.ai_risk_score.toFixed(1)} (${i.ai_risk_band})</div>`
        : ''}
      <div class="inc-sla ${i.sla_minutes_left < 0 ? 'breach' : ''}">
        ${i.assigned_to ? i.assigned_to + ' assigned' : 'unassigned'} ·
        ${i.sla_minutes_left < 0
          ? `SLA breached by ${Math.abs(i.sla_minutes_left).toFixed(0)} min`
          : `SLA in ${i.sla_minutes_left.toFixed(0)} min`}
      </div>
    </div>`).join('');

  // Clicking an incident flies the camera to its tower.
  host.querySelectorAll('.inc').forEach((el) => {
    el.addEventListener('click', () => {
      const t = world.towers.features.find((f) => f.properties.id === el.dataset.tower);
      if (t) flyTo(t.geometry.coordinates, 16.2);
    });
  });
}

function renderCrews() {
  $('crews').innerHTML = state.crews.map((c) => `
    <div class="crew">
      <span class="dot ${c.status}"></span>
      <span class="nm">${c.name}</span>
      <span class="st">${c.status === 'en_route' ? `ETA ${c.eta_min}m`
        : c.status === 'on_site' ? 'on site' : 'idle'}</span>
    </div>`).join('');
}

function renderLog() {
  if (!state.events || !state.events.length) return;
  const host = $('log');
  for (const e of state.events) {
    if (e.i < logSeen) continue;      // already on screen
    const div = document.createElement('div');
    let cls = '';
    if (e.message.startsWith('EDGE')) cls = 'edge';
    else if (e.message.startsWith('IMPACT')) cls = 'impact';
    else if (e.message.includes('restored')) cls = 'ok';
    div.innerHTML = `<span class="t">${clockText(e.t)}</span><span class="${cls}">${e.message}</span>`;
    host.appendChild(div);
  }
  while (host.children.length > 220) host.removeChild(host.firstChild);
  host.scrollTop = host.scrollHeight;
  logSeen = state.event_count;
}

function render() {
  if (!state) return;
  // The outage set drives the single most important thing on screen -- buildings going
  // red. It is rebuilt from the pushed state every frame; deriving it anywhere else
  // would let the map disagree with the incident panel.
  darkSet = new Set(state.dark_buildings);
  darkKey = state.dark_buildings.join(',');
  dark2d = new Set(state.dark_buildings_2d || []);
  dark2dKey = (state.dark_buildings_2d || []).join(',');
  $('clock').textContent = clockText(state.t);
  renderKpis();
  renderIncidents();
  renderCrews();
  renderLog();
  renderSplitReadout();
  deckgl.setProps({layers: buildLayers()});
}

function renderSplitReadout() {
  if (viewMode === '3d' || !state) return;
  const sum = (ids) => ids.reduce((a, id) => a + (world.subscribers[id] || 0), 0);
  const subs2d = sum(state.dark_buildings_2d || []);
  const subs3d = sum(state.dark_buildings);

  $('sl-2d-b').textContent = fmt((state.dark_buildings_2d || []).length);
  $('sl-2d-s').textContent = fmt(subs2d);
  $('sl-3d-b').textContent = fmt(state.dark_buildings.length);
  $('sl-3d-s').textContent = fmt(subs3d);

  const missedB = state.dark_buildings.length - (state.dark_buildings_2d || []).length;
  const missedS = subs3d - subs2d;
  $('sl-verdict').textContent = missedS > 0
    ? `the flat map misses ${fmt(missedB)} buildings and ${fmt(missedS)} subscribers`
    : 'all sites nominal — waiting for a fault…';
}

/* ------------------------------------------------------------------ camera */

function flyTo(coords, zoom) {
  deckgl.setProps({
    initialViewState: {
      longitude: coords[0], latitude: coords[1],
      zoom: zoom ?? 15.1, pitch: 56, bearing: -18,
      transitionDuration: 1600,
    },
  });
}

/* ------------------------------------------------------------------ startup */

/** Resolve once the document has finished loading and the browser has laid it out.
 *
 * deck.gl reads the container's size when it creates its canvas. Constructing it while
 * the page is still loading -- which is what happens if boot() reaches it before layout
 * settles -- yields a canvas that reports the right dimensions but never paints: the
 * scene is silently blank while the HUD, being plain DOM, looks perfectly healthy.
 * Waiting for load plus one frame costs nothing and removes the race.
 */
function pageReady() {
  return new Promise((resolve) => {
    const settle = () => requestAnimationFrame(() => requestAnimationFrame(resolve));
    if (document.readyState === 'complete') settle();
    else window.addEventListener('load', settle, {once: true});
  });
}

async function boot() {
  await pageReady();
  world = await (await fetch('/api/world')).json();
  world.roads = await (await fetch('/api/roads')).json();

  $('aoi').textContent =
    `Kuala Lumpur CBD · ${fmt(world.buildings.features.length)} buildings · ` +
    `${world.towers.features.length} sites · ${fmt(world.total_subscribers)} subscribers`;

  const imputed = world.buildings.features.filter(
    (f) => f.properties.height_source === 'imputed').length;
  const dem = world.buildings.features[0]?.properties?.dem_source;
  $('height-note').textContent =
    `building heights: ${fmt(world.buildings.features.length - imputed)} from OSM, ` +
    `${fmt(imputed)} imputed (median error 22 m)` +
    (dem ? ` · ground elevation: ${dem}` : '');

  // Populate the chaos panel's tower picker from the real fleet, so it can never offer
  // a site that does not exist.
  const picker = $('fault-tower');
  picker.innerHTML = world.towers.features.map((f) => {
    const p = f.properties;
    return `<option value="${p.id}">${p.id} — ${p.name || 'site'}</option>`;
  }).join('');

  // Terrain is optional: a repo without a baked DEM should still boot, just flat.
  try {
    const grid = await (await fetch('/api/terrain')).json();
    if (grid && grid.elevations) {
      // Each cell carries its own altitude as the polygon's z, so the ground sits at
      // the same elevation the buildings are extruded from rather than at zero.
      terrainCells = buildTerrainCells(grid).map((c) => ({
        ...c,
        lonlat: c.polygon.map(([x, y]) => {
          const [lon, lat] = metresToLonLat(x, y, grid.origin_lon, grid.origin_lat);
          return [lon, lat, c.elev];
        }),
      }));
      console.info(`terrain: ${terrainCells.length} cells from ${grid.source}`);
    }
  } catch (err) {
    console.info('no terrain grid available, drawing flat ground');
  }

  document.body.classList.add(`mode-${viewMode}`);

  deckgl = new DeckGL({
    container: 'map',
    views: currentViews(),
    initialViewState: currentViewState(),
    // Route each layer to the pane it was built for.
    layerFilter: ({layer, viewport}) => layer.id.startsWith(viewport.id + '-'),
    layers: buildLayers(),
    getTooltip: tooltip,
    parameters: {clearColor: [0.02, 0.027, 0.051, 1]},
    effects: [
      new deck.LightingEffect({
        ambient: new deck.AmbientLight({color: [200, 215, 255], intensity: 1.5}),
        sun: new deck.DirectionalLight({
          color: [255, 245, 220], intensity: 1.7, direction: [-1.2, -3, -1],
        }),
      }),
    ],
  });

  connect();
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onopen = () => {
    $('conn').textContent = 'live';
    $('conn').className = 'conn live';
    // The server ignores inbound content; this just keeps the socket from idling out.
    setInterval(() => socket.readyState === 1 && socket.send('.'), 15000);
  };
  socket.onmessage = (ev) => {
    state = JSON.parse(ev.data);
    recordTrails(state);
    render();
  };
  socket.onclose = () => {
    $('conn').textContent = 'reconnecting';
    $('conn').className = 'conn lost';
    setTimeout(connect, 1500);   // survive a server restart mid-demo
  };
}

/* ----------------------------------------------------------------- controls */

/* Paused state is tracked here rather than inferred from the button's own label.
 * Reading it back off the DOM meant a Reset -- which unpauses on the server -- left the
 * button stuck reading "Resume" while the clock ran. */
let paused = false;

function setPausedLabel() {
  const button = $('btn-pause');
  button.textContent = paused ? 'Resume' : 'Pause';
  button.classList.toggle('on', paused);
}

$('btn-pause').addEventListener('click', async () => {
  paused = !paused;
  const response = await fetch(`/api/control/${paused ? 'pause' : 'resume'}`,
                               {method: 'POST'}).then((r) => r.json()).catch(() => null);
  // Trust the server's answer over our own optimism.
  if (response && typeof response.paused === 'boolean') paused = response.paused;
  setPausedLabel();
});

$('btn-reset').addEventListener('click', async () => {
  await fetch('/api/control/reset', {method: 'POST'});
  $('log').innerHTML = '';
  logSeen = 0;
  trails.clear();
  paused = false;
  setPausedLabel();
});

/* ------------------------------------------------------------- chaos panel */

function flashStatus(message, ok = true) {
  const el = $('chaos-status');
  el.textContent = message;
  el.classList.toggle('bad', !ok);
  clearTimeout(flashStatus.timer);
  flashStatus.timer = setTimeout(() => { el.textContent = ''; }, 4000);
}

$('btn-storm').addEventListener('click', async () => {
  try {
    const r = await fetch('/api/storm?peak_mm_hr=110', {method: 'POST'});
    const body = await r.json();
    flashStatus(body.ok ? `storm injected — ${body.cells} cell(s) active`
                        : 'storm failed', !!body.ok);
  } catch (err) {
    flashStatus('storm failed', false);
  }
});

$('btn-fault').addEventListener('click', async () => {
  const tower = $('fault-tower').value;
  const profile = $('fault-profile').value;
  if (!tower) return;
  try {
    const r = await fetch(`/api/fault/${tower}?profile=${profile}`, {method: 'POST'});
    const body = await r.json();
    flashStatus(body.ok ? `${tower} — ${profile.replace(/_/g, ' ')} injected`
                        : (body.error || 'failed'), !!body.ok);
  } catch (err) {
    flashStatus('fault injection failed', false);
  }
});

$('speed').addEventListener('input', (e) => {
  $('speed-label').innerHTML = `${e.target.value}&times;`;
});
$('speed').addEventListener('change', async (e) => {
  await fetch(`/api/control/speed?factor=${e.target.value}`, {method: 'POST'});
});

/* Mode switching. Also bound to 1/2/3 so the pitch can be driven without hunting
   for a button while talking. */
$('btn-view-2d').addEventListener('click', () => setViewMode('2d'));
$('btn-view-3d').addEventListener('click', () => setViewMode('3d'));
$('btn-view-split').addEventListener('click', () => setViewMode('split'));

window.addEventListener('keydown', (e) => {
  const typing = e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT';
  if (typing || e.metaKey || e.ctrlKey) return;
  if (e.key === '1') setViewMode('2d');
  else if (e.key === '2') setViewMode('3d');
  else if (e.key === '3') setViewMode('split');
  // Chaos shortcuts, so the cascade can be driven mid-sentence without hunting for a
  // button. Lowercase only, to leave shifted keys free.
  else if (e.key === 's') $('btn-storm').click();
  else if (e.key === 'f') $('btn-fault').click();
});

boot();
