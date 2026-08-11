"""Site the simulated network: macro cells on rooftops, the way operators actually do it.

Towers are not scattered at random. Real macro cells go on tall buildings, spaced far
enough apart to avoid self-interference but close enough to overlap, so that losing one
site degrades service rather than deleting it. Both properties matter to the demo: the
overlap is what makes "which buildings actually go dark" a non-trivial question.

    python scripts/place_towers.py --n 15 --out data/towers.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twinsync.geo import LocalFrame  # noqa: E402

# A rooftop mast adds a few metres above the parapet.
MAST_HEIGHT_M = 6.0

# Sites are named after their host building where OSM knows one.
SITE_PREFIX = "KL"


def load_buildings(path: Path) -> tuple[list[dict], LocalFrame, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data["features"]

    lons = [c[0] for f in features for c in f["geometry"]["coordinates"][0]]
    lats = [c[1] for f in features for c in f["geometry"]["coordinates"][0]]
    frame = LocalFrame.from_bounds(min(lons), min(lats), max(lons), max(lats))

    centroids, heights = [], []
    for feature in features:
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        x, y = frame.to_xy(ring[:-1, 0], ring[:-1, 1])
        centroids.append([float(np.mean(x)), float(np.mean(y))])
        heights.append(float(feature["properties"]["height"]))

    return features, frame, np.array(centroids), np.array(heights)


def choose_sites(centroids: np.ndarray, heights: np.ndarray, areas: np.ndarray,
                 n: int, min_sep: float) -> list[int]:
    """Greedily take the tallest buildings that respect a minimum spacing.

    Relaxes the spacing if the AOI cannot supply enough well-separated sites, rather
    than silently returning fewer towers than asked for.
    """
    # Prefer tall buildings with a real roof to stand on -- a 200 m tall, 60 m^2
    # footprint is a mast, not a rooftop.
    score = heights * np.log1p(areas)
    order = np.argsort(score)[::-1]

    separation = min_sep
    while separation > 40.0:
        chosen: list[int] = []
        for index in order:
            if len(chosen) >= n:
                break
            if heights[index] < 15.0:
                continue    # too short to be a useful macro site
            if chosen:
                distances = np.hypot(*(centroids[chosen] - centroids[index]).T)
                if distances.min() < separation:
                    continue
            chosen.append(int(index))
        if len(chosen) >= n:
            return chosen[:n]
        separation *= 0.8
        print(f"  only {len(chosen)} sites at {separation / 0.8:.0f} m spacing, "
              f"relaxing to {separation:.0f} m")

    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Place simulated telecom towers.")
    parser.add_argument("--buildings", default="data/buildings.geojson", type=Path)
    parser.add_argument("--out", default="data/towers.geojson", type=Path)
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--min-sep", type=float, default=280.0,
                        help="minimum spacing between sites, metres")
    parser.add_argument("--range", dest="range_m", type=float, default=650.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    features, frame, centroids, heights = load_buildings(args.buildings)
    areas = np.array([float(f["properties"].get("area_m2") or 1.0) for f in features])
    print(f"{len(features)} candidate buildings, tallest {heights.max():.0f} m")

    chosen = choose_sites(centroids, heights, areas, args.n, args.min_sep)
    print(f"chose {len(chosen)} sites")

    towers = []
    for rank, index in enumerate(chosen, start=1):
        feature = features[index]
        props = feature["properties"]
        lon, lat = frame.to_lonlat(centroids[index][0], centroids[index][1])
        antenna = float(heights[index]) + MAST_HEIGHT_M

        towers.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [round(float(lon), 7),
                                                          round(float(lat), 7)]},
            "properties": {
                "id": f"{SITE_PREFIX}-{rank:02d}",
                "name": props.get("name") or f"Site {rank:02d}",
                "antenna_height": round(antenna, 1),
                "roof_height": round(float(heights[index]), 1),
                "range_m": args.range_m,
                "host_building": feature.get("id"),
            },
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"type": "FeatureCollection", "features": towers}), encoding="utf-8"
    )

    # Report the spacing actually achieved -- a site plan with two towers 30 m apart
    # would quietly ruin the overlap story.
    picked = centroids[chosen]
    pairwise = np.hypot(picked[:, None, 0] - picked[None, :, 0],
                        picked[:, None, 1] - picked[None, :, 1])
    np.fill_diagonal(pairwise, np.inf)

    print(f"\nwrote {args.out}")
    print(f"  antenna heights {min(t['properties']['antenna_height'] for t in towers):.0f}"
          f" - {max(t['properties']['antenna_height'] for t in towers):.0f} m")
    print(f"  nearest-neighbour spacing {pairwise.min():.0f} - "
          f"{pairwise.min(axis=1).max():.0f} m")
    for tower in towers:
        p = tower["properties"]
        print(f"    {p['id']}  {p['antenna_height']:6.1f} m  {p['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
