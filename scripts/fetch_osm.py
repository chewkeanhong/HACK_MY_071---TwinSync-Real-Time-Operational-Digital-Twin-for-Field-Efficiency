"""One-time OpenStreetMap fetch for the TwinSync digital twin.

Overpass is unreliable (observed 504s and read timeouts), so this script is built to
survive it: the AOI is split into tiles, every tile response is cached to disk, and a
re-run picks up exactly where the last one stopped. Mirrors rotate on failure.

Run this once, then commit data/. The demo itself never touches the network.

    python scripts/fetch_osm.py --bbox 3.140,101.705,3.162,101.725 --out data/
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import requests

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.osm.jp/api/interpreter",
]

HEADERS = {"User-Agent": "TwinSync/0.1 (ASEAN GeoAI Fusion 2026 prototype)"}

# Road classes a maintenance van can actually drive on.
DRIVABLE = (
    "motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    "living_street|service|motorway_link|trunk_link|primary_link|secondary_link|"
    "tertiary_link"
)

QUERIES = {
    "buildings": '[out:json][timeout:{t}];(way["building"]({bbox}););out geom;',
    "roads": '[out:json][timeout:{t}];(way["highway"~"^({classes})$"]({bbox}););out geom;',
    # Training labels only: buildings that already carry a height. Filtering server-side
    # keeps the response small enough to pull a much wider area without timing out,
    # which is the cheapest way to give the height model more to learn from.
    "building_labels": (
        '[out:json][timeout:{t}];('
        'way["building"]["height"]({bbox});'
        'way["building"]["building:levels"]({bbox});'
        ');out geom;'
    ),
}

# Which GeoJSON file each query kind produces.
OUTPUT_NAMES = {
    "buildings": "buildings.geojson",
    "roads": "roads.geojson",
    "building_labels": "train_buildings.geojson",
}

FEET_PER_METRE = 0.3048
DEFAULT_STOREY_HEIGHT = 3.2


# --------------------------------------------------------------------------- fetch


def tile_bbox(bbox: tuple[float, float, float, float], rows: int, cols: int):
    """Split (south, west, north, east) into rows*cols sub-boxes."""
    south, west, north, east = bbox
    dlat = (north - south) / rows
    dlon = (east - west) / cols
    for r in range(rows):
        for c in range(cols):
            yield r, c, (
                south + r * dlat,
                west + c * dlon,
                south + (r + 1) * dlat,
                west + (c + 1) * dlon,
            )


def run_query(query: str, timeout: int, max_attempts: int = 24) -> dict:
    """Execute an Overpass query, rotating mirrors with exponential backoff."""
    last_error = "no attempt made"
    for attempt in range(max_attempts):
        endpoint = MIRRORS[attempt % len(MIRRORS)]
        try:
            response = requests.post(
                endpoint, data={"data": query}, headers=HEADERS, timeout=timeout + 20
            )
            if response.status_code == 200:
                return response.json()
            # 429 = rate limited, 504 = gateway timeout: both are worth retrying.
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # network flake, JSON decode, whatever
            last_error = f"{type(exc).__name__}"

        delay = min(60.0, 2.0 * (2**attempt)) * (0.5 + random.random())
        host = endpoint.split("//")[1].split("/")[0]
        print(f"      retry {attempt + 1}/{max_attempts} ({host}: {last_error}) "
              f"sleeping {delay:.0f}s", flush=True)
        time.sleep(delay)

    raise RuntimeError(f"all mirrors failed after {max_attempts} attempts: {last_error}")


def fetch_tiles(kind: str, bbox, rows: int, cols: int, cache_dir: Path, timeout: int):
    """Fetch every tile for one feature kind, caching each to disk as it lands."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    elements: list[dict] = []
    tiles = list(tile_bbox(bbox, rows, cols))

    for index, (r, c, tb) in enumerate(tiles, start=1):
        cache_file = cache_dir / f"{kind}_r{r}c{c}.json"
        if cache_file.exists():
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"  [{index}/{len(tiles)}] {kind} r{r}c{c}: cached "
                  f"({len(payload.get('elements', []))} elements)", flush=True)
        else:
            bbox_str = ",".join(f"{v:.5f}" for v in tb)
            query = QUERIES[kind].format(t=timeout, bbox=bbox_str, classes=DRIVABLE)
            print(f"  [{index}/{len(tiles)}] {kind} r{r}c{c}: fetching...", flush=True)
            payload = run_query(query, timeout)
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            print(f"      got {len(payload.get('elements', []))} elements", flush=True)
            time.sleep(1.5)  # be a good citizen between tile requests

        elements.extend(payload.get("elements", []))

    return elements


# ------------------------------------------------------------------------ convert


def parse_height(tags: dict) -> tuple[float | None, str]:
    """Return (height_in_metres, source) from OSM tags. Source is 'osm' or 'none'."""
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        text = str(raw).strip().lower()
        try:
            if text.endswith("'"):  # feet, e.g. 120'
                return float(text.rstrip("'")) * FEET_PER_METRE, "osm"
            if "ft" in text:
                return float(text.replace("ft", "").strip()) * FEET_PER_METRE, "osm"
            value = float(text.replace("m", "").strip())
            if 1.0 <= value <= 900.0:
                return value, "osm"
        except ValueError:
            pass

    levels = tags.get("building:levels")
    if levels:
        try:
            value = float(str(levels).split(";")[0].strip()) * DEFAULT_STOREY_HEIGHT
            if 1.0 <= value <= 900.0:
                return value, "osm"
        except ValueError:
            pass

    return None, "none"


def ring_from_geometry(geometry: list[dict]) -> list[list[float]] | None:
    """Overpass 'out geom' gives [{lat, lon}, ...]; convert to a closed GeoJSON ring."""
    if not geometry or len(geometry) < 4:
        return None
    ring = [[round(p["lon"], 7), round(p["lat"], 7)] for p in geometry if "lon" in p]
    if len(ring) < 4:
        return None
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    if len(ring) < 4:
        return None
    return ring


def ring_area_m2(ring: list[list[float]]) -> float:
    """Shoelace area in m^2, using a local equirectangular projection."""
    lat0 = math.radians(sum(p[1] for p in ring) / len(ring))
    mx = 111320.0 * math.cos(lat0)
    my = 110540.0
    total = 0.0
    for i in range(len(ring) - 1):
        x1, y1 = ring[i][0] * mx, ring[i][1] * my
        x2, y2 = ring[i + 1][0] * mx, ring[i + 1][1] * my
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def buildings_to_geojson(elements: list[dict]) -> dict:
    features, seen = [], set()
    for element in elements:
        if element.get("type") != "way" or element["id"] in seen:
            continue
        ring = ring_from_geometry(element.get("geometry", []))
        if ring is None:
            continue
        area = ring_area_m2(ring)
        if area < 20.0:  # sheds, bin stores: noise in a city-scale twin
            continue

        seen.add(element["id"])
        tags = element.get("tags", {})
        height, source = parse_height(tags)
        features.append({
            "type": "Feature",
            "id": f"b{element['id']}",
            "properties": {
                "osm_id": element["id"],
                "name": tags.get("name"),
                "building": tags.get("building", "yes"),
                "amenity": tags.get("amenity"),
                "height": height,
                "height_source": source,
                "area_m2": round(area, 1),
            },
            "geometry": {"type": "Polygon", "coordinates": [ring]},
        })
    return {"type": "FeatureCollection", "features": features}


def roads_to_geojson(elements: list[dict]) -> dict:
    features, seen = [], set()
    for element in elements:
        if element.get("type") != "way" or element["id"] in seen:
            continue
        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        seen.add(element["id"])
        tags = element.get("tags", {})
        line = [[round(p["lon"], 7), round(p["lat"], 7)] for p in geometry if "lon" in p]
        features.append({
            "type": "Feature",
            "id": f"r{element['id']}",
            "properties": {
                "osm_id": element["id"],
                "name": tags.get("name"),
                "highway": tags.get("highway"),
                "oneway": tags.get("oneway", "no"),
                "maxspeed": tags.get("maxspeed"),
            },
            "geometry": {"type": "LineString", "coordinates": line},
        })
    return {"type": "FeatureCollection", "features": features}


# --------------------------------------------------------------------------- main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fetch OSM data for the TwinSync twin.")
    parser.add_argument("--bbox", default="3.140,101.705,3.162,101.725",
                        help="south,west,north,east")
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--kinds", default="buildings,roads",
                        help=f"comma-separated subset of {sorted(QUERIES)}")
    args = parser.parse_args(argv)

    bbox = tuple(float(v) for v in args.bbox.split(","))
    if len(bbox) != 4:
        parser.error("--bbox needs exactly 4 comma-separated numbers")

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    unknown = set(kinds) - set(QUERIES)
    if unknown:
        parser.error(f"unknown kind(s): {sorted(unknown)}")

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "raw"

    print(f"AOI {bbox}  ->  {args.rows}x{args.cols} tiles, kinds={kinds}\n")

    written = []
    for kind in kinds:
        print(kind.replace("_", " ").title())
        elements = fetch_tiles(kind, bbox, args.rows, args.cols, cache_dir, args.timeout)
        if kind == "roads":
            collection = roads_to_geojson(elements)
        else:
            collection = buildings_to_geojson(elements)

        path = out_dir / OUTPUT_NAMES[kind]
        path.write_text(json.dumps(collection), encoding="utf-8")
        written.append(path)

        features = collection["features"]
        print(f"\n  {kind}: {len(features)} features")
        if kind != "roads":
            tagged = sum(1 for f in features
                         if f["properties"]["height_source"] == "osm")
            print(f"    with height  {tagged} ({100 * tagged / max(len(features), 1):.1f}%)"
                  f" -> {len(features) - tagged} need imputation")
            tallest = sorted(
                ((f["properties"]["height"], f["properties"].get("name") or "unnamed")
                 for f in features if f["properties"]["height"]),
                reverse=True,
            )[:5]
            for height, name in tallest:
                print(f"      {height:6.1f} m  {name}")
        print()

    print("=" * 58)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
