"""Bake Copernicus DEM GLO-30 terrain into the repository.

The twin needs ground elevation for three things: Fresnel clearance over real terrain,
the Z coordinate in ST-DBSCAN fault localisation, and identifying low-lying road
segments that flood during a monsoon downpour. All three want the same thing -- a
height above sea level at an arbitrary (x, y) -- and none of them want a GDAL
dependency at demo time.

So this runs once, offline-prepares the answer, and commits it:

    python scripts/fetch_dem.py --data data

It reads the Copernicus DSM COG for the 1x1 degree tile covering the AOI straight out
of the public AWS open-data bucket (no credentials, no API key), resamples the AOI
window onto a regular grid in local metres, and writes:

    data/terrain.json        the grid itself, ~40 KB
    data/buildings.geojson   + ground_elev on every feature
    data/towers.geojson      + ground_elev on every feature

This mirrors what scripts/impute_heights.py already does with building heights: the
expensive computation happens once at build time and the result travels with the repo,
so the runtime side (twinsync/terrain.py) is pure numpy and the demo works on a
laptop in flight mode.

Copernicus DEM GLO-30 is a *surface* model (DSM), so over dense CBD blocks it includes
the buildings themselves. We are already modelling those explicitly from OSM
footprints, so double-counting them would inflate every rooftop. `--smooth` (on by
default) runs a morphological opening that pushes the surface back down towards street
level, which is the standard cheap DSM->DTM approximation. It is an approximation, and
it is recorded as one in the output metadata.

If the tile cannot be fetched -- no network, rasterio not installed, bucket layout
changed -- `--synthetic` generates a deterministic seeded surface instead. That path
stamps `dem_source: "synthetic"` through every output so it can never be mistaken for
real Copernicus data further down the pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.geo import LocalFrame  # noqa: E402

# Copernicus DEM GLO-30, public AWS open-data mirror. No credentials required.
# Tile naming is Copernicus_DSM_COG_10_{N|S}{lat:02d}_00_{E|W}{lon:03d}_00_DEM.
COP_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

# Grid resolution in metres. GLO-30 is nominally 30 m at the equator; sampling at 30 m
# keeps one grid cell per source pixel without pretending to detail we do not have.
DEFAULT_CELL_M = 30.0

# Pad the AOI so a ray leaving the built-up area still lands on real terrain.
AOI_PAD_M = 400.0

# DSM -> approximate DTM. Kernel radius in cells; 4 cells at 30 m is ~120 m, wider than
# a typical KL CBD block, so a building cluster gets pushed down to the streets around
# it while genuine hillsides survive.
OPENING_RADIUS_CELLS = 4


def tile_name(lon: float, lat: float) -> str:
    """Copernicus tile id for the 1x1 degree cell containing this coordinate."""
    lat_i, lon_i = math.floor(lat), math.floor(lon)
    ns = "N" if lat_i >= 0 else "S"
    ew = "E" if lon_i >= 0 else "W"
    return (f"Copernicus_DSM_COG_10_{ns}{abs(lat_i):02d}_00_"
            f"{ew}{abs(lon_i):03d}_00_DEM")


def tile_url(lon: float, lat: float) -> str:
    name = tile_name(lon, lat)
    return f"{COP_BUCKET}/{name}/{name}.tif"


# ------------------------------------------------------------------ real DEM


def sample_copernicus(lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
    """Read GLO-30 elevations at each (lon, lat) in the grid.

    Raises on any failure so the caller can decide whether to fall back to synthetic
    terrain -- silently substituting fake data for real data is exactly the kind of
    thing that should be loud.
    """
    import rasterio                                    # noqa: PLC0415
    from rasterio.errors import RasterioIOError        # noqa: PLC0415

    url = tile_url(float(lon_grid.mean()), float(lat_grid.mean()))
    print(f"  reading {url}")

    points = list(zip(lon_grid.ravel().tolist(), lat_grid.ravel().tolist()))
    try:
        with rasterio.open(url) as src:
            print(f"  tile crs={src.crs}, shape={src.shape}, nodata={src.nodata}")
            values = np.array([v[0] for v in src.sample(points)], dtype=np.float64)
            nodata = src.nodata
    except RasterioIOError as exc:
        raise RuntimeError(f"could not read {url}: {exc}") from exc

    if nodata is not None:
        values = np.where(values == nodata, np.nan, values)
    # GLO-30 encodes ocean/void as large negatives in some builds.
    values = np.where(values < -400.0, np.nan, values)

    grid = values.reshape(lon_grid.shape)
    if np.isnan(grid).all():
        raise RuntimeError("every sampled DEM pixel was nodata")
    if np.isnan(grid).any():
        grid = _fill_nan(grid)
    return grid


def _fill_nan(grid: np.ndarray) -> np.ndarray:
    """Replace voids with the grid mean. Voids over a city block are rare and small."""
    filled = grid.copy()
    mean = float(np.nanmean(filled))
    holes = int(np.isnan(filled).sum())
    print(f"  filling {holes} void cells with the tile mean ({mean:.1f} m)")
    filled[np.isnan(filled)] = mean
    return filled


def morphological_open(grid: np.ndarray, radius: int) -> np.ndarray:
    """Grey-scale opening (erode then dilate) with a square structuring element.

    Knocks the buildings out of a DSM without pulling down genuine terrain features
    wider than the kernel. Written with numpy shifts rather than scipy.ndimage so this
    stays runnable when only the base requirements are installed.
    """
    if radius <= 0:
        return grid

    def _slide(source: np.ndarray, reducer) -> np.ndarray:
        padded = np.pad(source, radius, mode="edge")
        stack = np.stack([
            padded[dy:dy + source.shape[0], dx:dx + source.shape[1]]
            for dy in range(2 * radius + 1)
            for dx in range(2 * radius + 1)
        ])
        return reducer(stack, axis=0)

    return _slide(_slide(grid, np.min), np.max)


# ------------------------------------------------------------- synthetic DEM


def synthetic_terrain(lon_grid: np.ndarray, lat_grid: np.ndarray,
                      seed: int = 42) -> np.ndarray:
    """A deterministic stand-in surface, used only when the real tile is unreachable.

    Shaped to be plausible for the Klang valley -- a broad river-valley low running
    through the middle with higher ground to the flanks -- because a flat plane would
    make every terrain-aware code path downstream untestable. It is *not* real data
    and everything it touches is stamped `dem_source: "synthetic"`.
    """
    rng = np.random.default_rng(seed)
    ny, nx = lon_grid.shape

    v = np.linspace(-1.0, 1.0, ny)[:, None]
    u = np.linspace(-1.0, 1.0, nx)[None, :]

    valley = 28.0 * (1.0 - np.exp(-((u * 1.6 + v * 0.4) ** 2) / 0.35))
    ridge = 14.0 * np.exp(-((u - 0.75) ** 2 + (v - 0.6) ** 2) / 0.5)
    tilt = 6.0 * v

    surface = 32.0 + valley + ridge + tilt
    for scale, amp in ((3, 3.0), (7, 1.4), (13, 0.6)):
        coarse = rng.normal(0.0, amp, size=(scale, scale))
        yi = np.linspace(0, scale - 1, ny)
        xi = np.linspace(0, scale - 1, nx)
        rows = np.array([np.interp(xi, np.arange(scale), coarse[r]) for r in range(scale)])
        surface += np.array([np.interp(yi, np.arange(scale), rows[:, c]) for c in range(nx)]).T
    return surface


# ------------------------------------------------------------------- driver


def build_grid(frame: LocalFrame, bounds_xy: tuple[float, float, float, float],
               cell_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Regular grid in local metres, plus the lon/lat of every node."""
    min_x, min_y, max_x, max_y = bounds_xy
    nx = int(math.ceil((max_x - min_x) / cell_m)) + 1
    ny = int(math.ceil((max_y - min_y) / cell_m)) + 1

    xs = min_x + np.arange(nx) * cell_m
    ys = min_y + np.arange(ny) * cell_m
    gx, gy = np.meshgrid(xs, ys)
    lon_grid, lat_grid = frame.to_lonlat(gx, gy)

    meta = {"min_x": min_x, "min_y": min_y, "nx": nx, "ny": ny, "cell_m": cell_m}
    return lon_grid, lat_grid, meta


def bilinear(grid: np.ndarray, meta: dict, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample the elevation grid at arbitrary local-metre coordinates.

    Duplicated deliberately from twinsync.terrain: this script must be able to stamp
    ground_elev without importing the runtime module it is generating input for.
    """
    fx = (np.asarray(x, dtype=np.float64) - meta["min_x"]) / meta["cell_m"]
    fy = (np.asarray(y, dtype=np.float64) - meta["min_y"]) / meta["cell_m"]
    fx = np.clip(fx, 0.0, meta["nx"] - 1.0000001)
    fy = np.clip(fy, 0.0, meta["ny"] - 1.0000001)

    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - x0, fy - y0
    x1, y1 = np.minimum(x0 + 1, meta["nx"] - 1), np.minimum(y0 + 1, meta["ny"] - 1)

    return ((1 - tx) * (1 - ty) * grid[y0, x0] + tx * (1 - ty) * grid[y0, x1]
            + (1 - tx) * ty * grid[y1, x0] + tx * ty * grid[y1, x1])


def ring_centre_lonlat(feature: dict) -> tuple[float, float]:
    geometry = feature["geometry"]
    if geometry["type"] == "Point":
        lon, lat = geometry["coordinates"]
        return float(lon), float(lat)
    ring = geometry["coordinates"][0]
    return (float(np.mean([c[0] for c in ring[:-1]])),
            float(np.mean([c[1] for c in ring[:-1]])))


def stamp(path: Path, frame: LocalFrame, grid: np.ndarray, meta: dict,
          source: str) -> int:
    """Write ground_elev onto every feature of a GeoJSON file, in place."""
    if not path.exists():
        print(f"  skip {path.name} (not present)")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not features:
        return 0

    lons, lats = zip(*(ring_centre_lonlat(f) for f in features))
    x, y = frame.to_xy(np.array(lons), np.array(lats))
    elevations = bilinear(grid, meta, x, y)

    for feature, elevation in zip(features, elevations):
        feature["properties"]["ground_elev"] = round(float(elevation), 2)
        feature["properties"]["dem_source"] = source

    path.write_text(json.dumps(data), encoding="utf-8")
    print(f"  stamped {len(features)} features in {path.name} "
          f"({elevations.min():.1f}-{elevations.max():.1f} m)")
    return len(features)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bake Copernicus DEM into the repo.")
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default=None, type=Path,
                        help="terrain grid path (default: <data>/terrain.json)")
    parser.add_argument("--cell", type=float, default=DEFAULT_CELL_M,
                        help="grid resolution in metres")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip the download and generate a seeded surface")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-smooth", dest="smooth", action="store_false",
                        help="keep the raw DSM instead of approximating a DTM")
    args = parser.parse_args(argv)

    out = args.out or (args.data / "terrain.json")
    buildings_path = args.data / "buildings.geojson"
    if not buildings_path.exists():
        parser.error(f"{buildings_path} not found -- run scripts/fetch_osm.py first")

    # The AOI comes from the buildings, so the grid always covers the world the
    # simulation actually loads.
    raw = json.loads(buildings_path.read_text(encoding="utf-8"))
    all_lon = [c[0] for f in raw["features"] for c in f["geometry"]["coordinates"][0]]
    all_lat = [c[1] for f in raw["features"] for c in f["geometry"]["coordinates"][0]]
    frame = LocalFrame.from_bounds(min(all_lon), min(all_lat), max(all_lon), max(all_lat))

    xs, ys = frame.to_xy(np.array(all_lon), np.array(all_lat))
    bounds = (float(xs.min()) - AOI_PAD_M, float(ys.min()) - AOI_PAD_M,
              float(xs.max()) + AOI_PAD_M, float(ys.max()) + AOI_PAD_M)

    lon_grid, lat_grid, meta = build_grid(frame, bounds, args.cell)
    span_x = (meta["nx"] - 1) * args.cell / 1000.0
    span_y = (meta["ny"] - 1) * args.cell / 1000.0
    print(f"grid: {meta['nx']}x{meta['ny']} @ {args.cell:.0f} m "
          f"({span_x:.1f} x {span_y:.1f} km)")

    source = "synthetic"
    smoothed = False
    if args.synthetic:
        print("using synthetic terrain (--synthetic)")
        grid = synthetic_terrain(lon_grid, lat_grid, seed=args.seed)
    else:
        print(f"fetching Copernicus GLO-30 tile "
              f"{tile_name(float(lon_grid.mean()), float(lat_grid.mean()))}")
        try:
            grid = sample_copernicus(lon_grid, lat_grid)
            source = "copernicus-glo30"
            if args.smooth:
                opened = morphological_open(grid, OPENING_RADIUS_CELLS)
                drop = float((grid - opened).mean())
                print(f"  DSM->DTM opening: mean surface lowered {drop:.1f} m")
                grid = opened
                smoothed = True
        except Exception as exc:                          # noqa: BLE001
            print(f"  FAILED: {exc}")
            print("  falling back to synthetic terrain -- outputs will be labelled "
                  "dem_source=synthetic")
            grid = synthetic_terrain(lon_grid, lat_grid, seed=args.seed)

    payload = {
        "source": source,
        "tile": tile_name(float(lon_grid.mean()), float(lat_grid.mean())),
        "smoothed_dsm_to_dtm": smoothed,
        "opening_radius_cells": OPENING_RADIUS_CELLS if smoothed else 0,
        "origin_lon": frame.origin_lon,
        "origin_lat": frame.origin_lat,
        "min_x": round(meta["min_x"], 3),
        "min_y": round(meta["min_y"], 3),
        "nx": meta["nx"],
        "ny": meta["ny"],
        "cell_m": meta["cell_m"],
        "min_elev": round(float(grid.min()), 2),
        "max_elev": round(float(grid.max()), 2),
        # Row-major, y ascending north. Centimetre precision is well beyond GLO-30's
        # real accuracy but keeps the file exactly reproducible.
        "elevations": [round(float(v), 2) for v in grid.ravel()],
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    size_kb = out.stat().st_size / 1024.0
    print(f"\nwrote {out} ({size_kb:.0f} KB, source={source})")
    print(f"  elevation range {payload['min_elev']:.1f} - {payload['max_elev']:.1f} m")

    print("\nstamping ground_elev:")
    stamp(buildings_path, frame, grid, meta, source)
    stamp(args.data / "towers.geojson", frame, grid, meta, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
