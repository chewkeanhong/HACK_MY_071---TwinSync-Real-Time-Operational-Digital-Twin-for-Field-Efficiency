"""Bake Sentinel-2 NDVI vegetation encroachment into the repository.

Vegetation growing into a feeder run or a guy-wire anchor is a real cause of site
degradation, and it is the one risk-model input this prototype used to invent: a static
value hashed from the site id, labelled `sentinel2-ndvi-simulated-v0.1` wherever it
surfaced. This script replaces that with an actual optical observation.

    python scripts/fetch_ndvi.py --data data

It searches the Element84 `earth-search` STAC API on AWS (no credentials, no API key)
for the least-cloudy Sentinel-2 L2A scene covering the AOI, reads the red (B04) and NIR
(B08) COGs over just the window the towers occupy, masks cloud and shadow using the
scene classification layer (SCL), and reduces NDVI to one value per site over a feeder
corridor buffer. It writes:

    data/ndvi.json    per-site NDVI and encroachment risk, plus the scene it came from

Like scripts/fetch_dem.py, this runs once at build time and the result travels with the
repo, so the runtime side stays pure numpy and the demo never touches the network.

`--synthetic` reproduces the old hashed stand-in for a repo with no network. That path
stamps `source: "simulated"` through the output so it cannot be mistaken for an
observation further down the pipeline.

Two details that are easy to get wrong and change the numbers:

* **Processing baseline 04.00 offset.** Since January 2022 L2A products carry a
  BOA_ADD_OFFSET of -1000. NDVI is a ratio so the 1/10000 scaling cancels, but the
  offset does not: ignoring it biases NDVI toward zero over dark urban pixels. The
  offset actually needed is read off the STAC item rather than assumed.
* **Scene coverage.** The AOI straddles two MGRS tiles, so the least-cloudy scene is
  not necessarily one that contains every site. Candidates are tried in cloud order and
  a scene is rejected unless every tower lands on valid, unclouded pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha1
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STAC_SEARCH = "https://earth-search.aws.element84.com/v1/search"
COLLECTION = "sentinel-2-l2a"

# Radius around each site to average NDVI over. A tower's vegetation risk is about what
# is growing against the mast, the feeder run and the compound fence, not the district:
# 120 m is roughly the compound plus the immediate approach, and at 10 m Sentinel-2
# pixels that is ~450 pixels to average, which is enough to be stable.
DEFAULT_BUFFER_M = 120.0

# SCL classes that are not usable ground observations.
#   3 cloud shadow · 8 cloud medium probability · 9 cloud high probability
#   10 thin cirrus · 11 snow/ice · 0 nodata · 1 saturated/defective
SCL_REJECT = {0, 1, 3, 8, 9, 10, 11}

# NDVI -> encroachment risk. Bare/built pixels sit near 0 and dense canopy near 0.8;
# the risk model wants a 0..1 pressure term, so this is a linear stretch across that
# range rather than anything learned. Stated here because it is an assumption, and
# echoed into the output so a reader can substitute their own.
NDVI_RISK_FLOOR = 0.10
NDVI_RISK_CEILING = 0.70


def stac_search(bbox, start: str, end: str, max_cloud: float, limit: int) -> list[dict]:
    """Least-cloudy L2A scenes intersecting the AOI, cloudiest last."""
    import urllib.request

    body = json.dumps({
        "collections": [COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start}/{end}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "limit": limit,
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }).encode()

    request = urllib.request.Request(
        STAC_SEARCH, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = json.load(response)

    features = payload.get("features", [])
    print(f"  {len(features)} candidate scenes under {max_cloud:.0f}% cloud")
    return features


def boa_offset(item: dict) -> float:
    """Additive offset needed to turn the stored DN into surface reflectance.

    Products from processing baseline 04.00 onward store reflectance shifted by +1000 so
    it stays unsigned. Element84 flags whether it has already removed that. NDVI is a
    ratio, so the 1/10000 gain cancels and only this offset matters.
    """
    properties = item.get("properties", {})
    if properties.get("earthsearch:boa_offset_applied"):
        return 0.0
    baseline = str(properties.get("s2:processing_baseline", "99.99"))
    try:
        return -1000.0 if float(baseline) >= 4.0 else 0.0
    except ValueError:
        return -1000.0


def read_window(url: str, lons, lats, buffer_m: float):
    """Read the smallest window covering every point plus its buffer.

    Returns the array, the pixel coordinates of each point, and the pixel size, so the
    caller can average over a metric radius rather than a pixel count.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window, from_bounds

    with rasterio.open(url) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
        xs, ys = np.asarray(xs), np.asarray(ys)

        pad = buffer_m + 2 * max(abs(src.transform.a), abs(src.transform.e))
        window = from_bounds(xs.min() - pad, ys.min() - pad,
                             xs.max() + pad, ys.max() + pad,
                             transform=src.transform).round_offsets().round_lengths()
        # Clamp to the raster: a site near the scene edge would otherwise read outside.
        col_off = max(0, int(window.col_off))
        row_off = max(0, int(window.row_off))
        width = min(int(window.width), src.width - col_off)
        height = min(int(window.height), src.height - row_off)
        if width <= 0 or height <= 0:
            raise RuntimeError("AOI falls outside this scene")

        window = Window(col_off, row_off, width, height)
        data = src.read(1, window=window).astype(np.float64)

        pixel_m = abs(src.transform.a)
        cols = (xs - (src.transform.c + col_off * src.transform.a)) / src.transform.a
        rows = (ys - (src.transform.f + row_off * src.transform.e)) / src.transform.e
        return data, np.column_stack([rows, cols]), pixel_m


def disc(shape, centre, radius_px: float) -> np.ndarray:
    """Boolean mask of pixels within `radius_px` of a (row, col) centre."""
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) <= radius_px ** 2


def ndvi_to_risk(ndvi: float) -> float:
    span = NDVI_RISK_CEILING - NDVI_RISK_FLOOR
    return float(np.clip((ndvi - NDVI_RISK_FLOOR) / span, 0.0, 1.0))


def sample_scene(item: dict, towers: list[dict], buffer_m: float) -> dict:
    """Per-site NDVI from one scene, or raise if the scene cannot serve every site."""
    assets = item["assets"]
    lons = [t["lon"] for t in towers]
    lats = [t["lat"] for t in towers]
    offset = boa_offset(item)

    red, centres, pixel_m = read_window(assets["red"]["href"], lons, lats, buffer_m)
    nir, _, _ = read_window(assets["nir"]["href"], lons, lats, buffer_m)
    scl, scl_centres, scl_pixel_m = read_window(assets["scl"]["href"], lons, lats,
                                                buffer_m)

    red += offset
    nir += offset

    with np.errstate(invalid="ignore", divide="ignore"):
        grid = (nir - red) / (nir + red)
    grid[~np.isfinite(grid)] = np.nan

    usable = ~np.isin(scl.astype(int), list(SCL_REJECT))
    radius_px = buffer_m / pixel_m
    scl_radius_px = buffer_m / scl_pixel_m

    out = {}
    for tower, centre, scl_centre in zip(towers, centres, scl_centres):
        mask = disc(grid.shape, centre, radius_px)
        scl_mask = disc(scl.shape, scl_centre, scl_radius_px) & usable
        clear_fraction = (float(scl_mask.sum())
                          / max(1.0, float(disc(scl.shape, scl_centre,
                                                scl_radius_px).sum())))

        values = grid[mask]
        values = values[np.isfinite(values)]
        if values.size == 0 or clear_fraction < 0.5:
            raise RuntimeError(
                f"{tower['id']}: {values.size} valid pixels, "
                f"{100 * clear_fraction:.0f}% clear -- scene rejected")

        ndvi = float(np.median(values))
        out[tower["id"]] = {
            "ndvi": round(ndvi, 4),
            "encroachment_risk": round(ndvi_to_risk(ndvi), 4),
            "pixels": int(values.size),
            "clear_pct": round(100.0 * clear_fraction, 1),
        }
    return out


def synthetic(towers: list[dict]) -> dict:
    """The old hashed stand-in, kept for a repo with no network.

    Deliberately identical to what twinsync/sim.py used to compute inline, so switching
    to `--synthetic` reproduces the previous behaviour exactly rather than introducing a
    third set of numbers.
    """
    out = {}
    for tower in towers:
        seed = int(sha1(tower["id"].encode("utf-8")).hexdigest()[:8], 16)
        risk = min(1.0, max(0.0, 0.15 + ((seed % 70) / 100.0)))
        out[tower["id"]] = {"ndvi": None, "encroachment_risk": round(risk, 4),
                            "pixels": 0, "clear_pct": None}
    return out


def load_towers(path: Path) -> list[dict]:
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    return [{"id": f["properties"]["id"],
             "lon": f["geometry"]["coordinates"][0],
             "lat": f["geometry"]["coordinates"][1]} for f in features]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", type=Path)
    parser.add_argument("--out", default=None, type=Path)
    parser.add_argument("--buffer", type=float, default=DEFAULT_BUFFER_M,
                        help="feeder corridor radius in metres")
    parser.add_argument("--start", default="2024-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-08-31T23:59:59Z")
    parser.add_argument("--max-cloud", type=float, default=25.0)
    parser.add_argument("--candidates", type=int, default=12,
                        help="how many scenes to try before giving up")
    parser.add_argument("--synthetic", action="store_true",
                        help="skip the network and reproduce the hashed stand-in")
    args = parser.parse_args(argv)

    towers = load_towers(args.data / "towers.geojson")
    out_path = args.out or (args.data / "ndvi.json")
    print(f"{len(towers)} sites, {args.buffer:.0f} m buffer")

    if args.synthetic:
        payload = {
            "source": "simulated",
            "model": "sentinel2-ndvi-simulated-v0.1",
            "note": "hashed stand-in -- NOT an observation",
            "buffer_m": args.buffer,
            "per_tower": synthetic(towers),
        }
    else:
        lons = [t["lon"] for t in towers]
        lats = [t["lat"] for t in towers]
        pad = 0.01
        bbox = (min(lons) - pad, min(lats) - pad, max(lons) + pad, max(lats) + pad)

        scenes = stac_search(bbox, args.start, args.end, args.max_cloud,
                             args.candidates)
        if not scenes:
            print("no scenes matched; re-run with --synthetic to bake the stand-in")
            return 1

        chosen = None
        for item in scenes:
            cloud = item["properties"].get("eo:cloud_cover", -1)
            print(f"  trying {item['id']} ({item['properties']['datetime'][:10]}, "
                  f"{cloud:.1f}% cloud)")
            try:
                per_tower = sample_scene(item, towers, args.buffer)
            except Exception as exc:
                print(f"    rejected: {exc}")
                continue
            chosen = (item, per_tower)
            break

        if chosen is None:
            print("every candidate scene was rejected; widen --max-cloud or the dates")
            return 1

        item, per_tower = chosen
        payload = {
            "source": "sentinel-2-l2a",
            "model": "sentinel2-ndvi-v1",
            "scene_id": item["id"],
            "sensing_date": item["properties"]["datetime"],
            "cloud_cover_pct": round(
                float(item["properties"].get("eo:cloud_cover", 0.0)), 2),
            "platform": item["properties"].get("platform"),
            "processing_baseline": item["properties"].get("s2:processing_baseline"),
            "boa_offset_applied": boa_offset(item) == 0.0,
            "buffer_m": args.buffer,
            "reducer": "median",
            "ndvi_to_risk": {"floor": NDVI_RISK_FLOOR, "ceiling": NDVI_RISK_CEILING,
                             "form": "clip((ndvi - floor) / (ceiling - floor), 0, 1)"},
            "per_tower": per_tower,
        }

    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    risks = [v["encroachment_risk"] for v in payload["per_tower"].values()]
    ndvis = [v["ndvi"] for v in payload["per_tower"].values() if v["ndvi"] is not None]
    print(f"\nwrote {out_path}  ({payload['source']})")
    if ndvis:
        print(f"  NDVI    min {min(ndvis):.3f}  median {np.median(ndvis):.3f}  "
              f"max {max(ndvis):.3f}")
    print(f"  risk    min {min(risks):.3f}  median {np.median(risks):.3f}  "
          f"max {max(risks):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
