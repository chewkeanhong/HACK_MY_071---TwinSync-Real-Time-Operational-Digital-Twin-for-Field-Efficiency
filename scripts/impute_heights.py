"""Fill in the building heights OpenStreetMap does not have.

Only about a sixth of the footprints in the KL CBD carry a `height` or
`building:levels` tag. A 3D twin with holes in it is useless -- and, worse, line-of-sight
modelling silently *under*-predicts blockage when buildings are missing, so the coverage
story would be wrong in the direction that looks safe.

So: train on the footprints that do carry a height, predict the rest. Every prediction is
marked `height_source: "imputed"` so the dashboard can shade them differently and nobody
mistakes a guess for a survey.

A note on honesty, because this number goes in the pitch: the neighbourhood features
(mean/max height of nearby buildings) MUST be computed from training labels only. Deriving
them from all labelled buildings leaks the test set and flatters the score by several
metres. :class:`HeightImputer` keeps that boundary explicit.

    python scripts/impute_heights.py --in data/buildings.geojson --report
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

# Physically implausible predictions are worse than useless in a demo -- clamp.
MIN_HEIGHT = 3.0
MAX_HEIGHT = 460.0

# Roughly how tall each building class tends to be in a dense Asian CBD.
KIND_RANK = {
    "shed": 0, "hut": 0, "garage": 0, "garages": 0, "carport": 0, "roof": 0,
    "kiosk": 0, "service": 0, "house": 1, "bungalow": 1, "detached": 1,
    "terrace": 1, "retail": 2, "commercial": 3, "industrial": 2, "warehouse": 2,
    "school": 2, "civic": 2, "government": 3, "public": 2, "church": 2,
    "mosque": 2, "temple": 2, "parking": 3, "hospital": 4, "hotel": 5,
    "office": 5, "apartments": 5, "residential": 4, "dormitory": 4,
    "tower": 6, "skyscraper": 6, "train_station": 3, "yes": 2,
}
DEFAULT_KIND_RANK = 2

STATIC_FEATURES = [
    "log_area", "log_perimeter", "compactness", "vertices",
    "kind_rank", "has_amenity", "has_name", "dist_to_centre",
]
NEIGHBOUR_K = (3, 5, 10, 20)
MAX_K = max(NEIGHBOUR_K)
NEIGHBOUR_RADII = (100.0, 200.0, 400.0)


def _neighbour_feature_names() -> list[str]:
    names = []
    for k in NEIGHBOUR_K:
        names += [f"knn{k}_mean", f"knn{k}_max"]
    for r in NEIGHBOUR_RADII:
        names += [f"r{int(r)}_mean", f"r{int(r)}_count"]
    names += ["idw_mean", "nearest_dist"]
    return names


FEATURE_NAMES = STATIC_FEATURES + _neighbour_feature_names()


# ------------------------------------------------------------------ geometry


def local_scale(lat: float) -> tuple[float, float]:
    """Metres per degree of longitude/latitude at this latitude."""
    return 111_320.0 * math.cos(math.radians(lat)), 110_540.0


def static_features(features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Label-independent features plus centroids, in local metres."""
    lat0 = float(np.mean([f["geometry"]["coordinates"][0][0][1] for f in features]))
    mx, my = local_scale(lat0)

    rows, centroids = [], []
    for feature in features:
        props = feature["properties"]
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=np.float64)
        x, y = ring[:, 0] * mx, ring[:, 1] * my

        area = 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))
        perimeter = float(np.hypot(np.diff(x), np.diff(y)).sum())
        # Compactness: 1.0 for a circle, lower for sprawling or complex outlines.
        compactness = (4 * math.pi * area / perimeter**2) if perimeter > 0 else 0.0

        rows.append([
            math.log1p(area),
            math.log1p(perimeter),
            compactness,
            len(ring) - 1,
            KIND_RANK.get(props.get("building", "yes"), DEFAULT_KIND_RANK),
            1.0 if props.get("amenity") else 0.0,
            1.0 if props.get("name") else 0.0,   # named buildings skew notable/tall
            0.0,                                  # dist_to_centre, filled in below
        ])
        centroids.append((float(x[:-1].mean()), float(y[:-1].mean())))

    X = np.array(rows, dtype=np.float64)
    centroids = np.array(centroids, dtype=np.float64)
    centre = centroids.mean(axis=0)
    X[:, STATIC_FEATURES.index("dist_to_centre")] = np.hypot(*(centroids - centre).T)
    return X, centroids


# ------------------------------------------------------------------- model


class HeightImputer:
    """Predicts building height from footprint shape, class, and neighbourhood.

    The neighbourhood block is the interesting part: buildings cluster by height, so
    "how tall are the labelled buildings near me" carries more signal than any property
    of the footprint itself. It is computed strictly against the fitted training set,
    which is what keeps :meth:`cross_validate` honest.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1
        )
        self._train_centroids: np.ndarray | None = None
        self._train_heights: np.ndarray | None = None
        self._tree: cKDTree | None = None

    def _neighbour_features(self, query: np.ndarray, *, exclude_self: bool) -> np.ndarray:
        """Neighbourhood height statistics, computed only against the training set.

        A KD-tree rather than a full pairwise matrix: with ~15,000 training buildings a
        dense distance matrix is 1.8 GB, and it only grows if the training area widens.
        """
        train_h = self._train_heights
        tree = self._tree

        # One extra neighbour so a training row can discard itself.
        k = min(MAX_K + (1 if exclude_self else 0), len(train_h))
        distances, indices = tree.query(query, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        if exclude_self:
            distances, indices = distances[:, 1:], indices[:, 1:]

        heights = train_h[indices]
        columns = []
        available = heights.shape[1]
        for kk in NEIGHBOUR_K:
            take = max(1, min(kk, available))
            columns.append(heights[:, :take].mean(axis=1))
            columns.append(heights[:, :take].max(axis=1))

        # Radius statistics derived from the same k nearest, which is exact whenever
        # fewer than k neighbours fall inside the radius -- the usual case at 100-400 m.
        for radius in NEIGHBOUR_RADII:
            inside = distances <= radius
            count = inside.sum(axis=1)
            total = np.where(inside, heights, 0.0).sum(axis=1)
            columns.append(np.where(count > 0, total / np.maximum(count, 1),
                                    heights[:, 0]))
            columns.append(count.astype(np.float64))

        # Inverse-distance weighting over the local neighbourhood only; weighting by the
        # whole city just re-derives the global mean.
        weights = 1.0 / np.maximum(distances, 1.0) ** 1.5
        columns.append((weights * heights).sum(axis=1) / weights.sum(axis=1))
        columns.append(distances[:, 0])
        return np.column_stack(columns)

    def fit(self, static: np.ndarray, centroids: np.ndarray, heights: np.ndarray) -> "HeightImputer":
        self._train_centroids = centroids
        self._train_heights = heights
        self._tree = cKDTree(centroids)
        neighbours = self._neighbour_features(centroids, exclude_self=True)
        # Train in log space: heights span 3 m to 450 m and the tail would otherwise
        # dominate the loss.
        self.model.fit(np.column_stack([static, neighbours]), np.log1p(heights))
        return self

    def predict(self, static: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        neighbours = self._neighbour_features(centroids, exclude_self=False)
        predicted = np.expm1(self.model.predict(np.column_stack([static, neighbours])))
        return np.clip(predicted, MIN_HEIGHT, MAX_HEIGHT)

    def cross_validate(self, static, centroids, heights, *,
                       extra: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                       folds=5, repeats=3) -> dict:
        """Repeated K-fold over the *scene* buildings, which is the real task.

        ``extra`` (the wider-area labels) sits in the training fold every time and is
        never scored. That matters: the wider area is mostly rows of identical terraced
        houses whose neighbours share their exact height, so folding them into the test
        set produces a median error of 0.00 m and an MAE that says nothing about the
        dense CBD high-rise this twin is actually built on. Scoring only the held-out
        CBD buildings measures the question being asked at deployment -- given every
        label we could obtain, how wrong are we about a downtown building?
        """
        mae, medae, r2 = [], [], []
        for seed in range(repeats):
            splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
            for train_idx, test_idx in splitter.split(static):
                train_static = static[train_idx]
                train_centroids = centroids[train_idx]
                train_heights = heights[train_idx]
                if extra is not None:
                    train_static = np.vstack([train_static, extra[0]])
                    train_centroids = np.vstack([train_centroids, extra[1]])
                    train_heights = np.concatenate([train_heights, extra[2]])

                fold = HeightImputer(seed=self.seed)
                fold.fit(train_static, train_centroids, train_heights)
                predicted = fold.predict(static[test_idx], centroids[test_idx])
                actual = heights[test_idx]
                mae.append(mean_absolute_error(actual, predicted))
                medae.append(float(np.median(np.abs(actual - predicted))))
                r2.append(r2_score(actual, predicted))
        return {"mae": float(np.mean(mae)), "medae": float(np.mean(medae)),
                "r2": float(np.mean(r2))}


# -------------------------------------------------------------------- main


def load_labelled(path: Path) -> list[dict]:
    """Buildings whose height came from OSM, not from a previous run of this script."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [f for f in data["features"]
            if f["properties"].get("height_source") == "osm"
            and f["properties"].get("height") is not None]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Impute missing OSM building heights.")
    parser.add_argument("--in", dest="infile", default="data/buildings.geojson", type=Path)
    parser.add_argument("--out", dest="outfile", default=None, type=Path,
                        help="defaults to overwriting --in")
    parser.add_argument("--train-extra", type=Path, default=None,
                        help="extra labelled buildings (e.g. a wider AOI) used for "
                             "training only, never written to the output")
    parser.add_argument("--report", action="store_true", help="print cross-validated accuracy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    out_path = args.outfile or args.infile
    data = json.loads(args.infile.read_text(encoding="utf-8"))
    scene = data["features"]
    print(f"loaded {len(scene)} buildings from {args.infile}")

    # Ground truth is "OSM told us" -- never a previous run's own predictions.
    surveyed = [f for f in scene if f["properties"].get("height_source") == "osm"
                and f["properties"].get("height") is not None]
    training_set = list(surveyed)

    if args.train_extra:
        # The wider training area geographically CONTAINS the scene, so every surveyed
        # building in the scene also appears in the extra file. Left in, each one lands
        # in both the training and test folds at distance zero, the neighbour features
        # read its own height straight back, and cross-validation reports a median error
        # of exactly 0.00 m. Deduplicate by OSM id or the score is fiction.
        scene_ids = {f["properties"].get("osm_id") for f in scene}
        extra_all = load_labelled(args.train_extra)
        extra = [f for f in extra_all
                 if f["properties"].get("osm_id") not in scene_ids]
        dropped = len(extra_all) - len(extra)
        training_set += extra
        print(f"  + {len(extra)} extra labelled buildings from {args.train_extra}"
              f" ({dropped} duplicates of scene buildings dropped)")

    print(f"  {len(surveyed)} surveyed in scene "
          f"({100 * len(surveyed) / len(scene):.1f}%), "
          f"{len(scene) - len(surveyed)} to impute, "
          f"{len(training_set)} training rows")

    if len(training_set) < 30:
        print("ERROR: not enough labelled buildings to train on", file=sys.stderr)
        return 1

    # Features must be computed in one coordinate frame, so stack scene + extras and
    # slice afterwards.
    combined = scene + training_set[len(surveyed):]
    static_all, centroids_all = static_features(combined)
    static_scene, centroids_scene = static_all[:len(scene)], centroids_all[:len(scene)]

    surveyed_idx = np.array([i for i, f in enumerate(scene)
                             if f["properties"].get("height_source") == "osm"
                             and f["properties"].get("height") is not None])
    extra_idx = np.arange(len(scene), len(combined))
    train_idx = np.concatenate([surveyed_idx, extra_idx]).astype(int)

    static_train = static_all[train_idx]
    centroids_train = centroids_all[train_idx]
    heights_train = np.array([combined[i]["properties"]["height"] for i in train_idx],
                             dtype=np.float64)

    imputer = HeightImputer(seed=args.seed)

    if args.report:
        # Score only the buildings in the scene; the wider-area labels are training aid.
        n_scene = len(surveyed_idx)
        scene_static = static_all[surveyed_idx]
        scene_centroids = centroids_all[surveyed_idx]
        scene_heights = np.array([scene[i]["properties"]["height"] for i in surveyed_idx],
                                 dtype=np.float64)
        extra_block = None
        if len(extra_idx):
            extra_block = (static_all[extra_idx], centroids_all[extra_idx],
                           np.array([combined[i]["properties"]["height"]
                                     for i in extra_idx], dtype=np.float64))

        scores = imputer.cross_validate(scene_static, scene_centroids, scene_heights,
                                        extra=extra_block)
        median_guess = float(np.mean(np.abs(scene_heights - np.median(scene_heights))))
        print(f"\n  cross-validated on the {n_scene} surveyed CBD buildings "
              f"(5-fold x 3 repeats)")
        if extra_block is not None:
            print(f"    with {len(extra_idx):,} wider-area labels always in the "
                  f"training fold, never scored")
        print(f"    MAE            {scores['mae']:6.2f} m")
        print(f"    median abs err {scores['medae']:6.2f} m   <- half of buildings beat this")
        print(f"    R^2            {scores['r2']:6.3f}")
        print(f"    median guess   {median_guess:6.2f} m MAE  "
              f"-> {100 * (1 - scores['mae'] / median_guess):.0f}% better than guessing")

    imputer.fit(static_train, centroids_train, heights_train)

    if args.report:
        order = np.argsort(imputer.model.feature_importances_)[::-1]
        print("\n  feature importance")
        for i in order[:6]:
            print(f"    {FEATURE_NAMES[i]:<18} {imputer.model.feature_importances_[i]:.3f}")

    missing = np.array([i for i, f in enumerate(scene)
                        if f["properties"].get("height_source") != "osm"
                        or f["properties"].get("height") is None])
    if len(missing):
        predicted = imputer.predict(static_scene[missing], centroids_scene[missing])
        for idx, height in zip(missing, predicted):
            scene[idx]["properties"]["height"] = round(float(height), 1)
            scene[idx]["properties"]["height_source"] = "imputed"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data), encoding="utf-8")

    heights = np.array([f["properties"]["height"] for f in scene])
    print(f"\nwrote {out_path}")
    print(f"  heights present  {len(scene)}/{len(scene)}")
    print(f"  median {np.median(heights):.1f} m | mean {heights.mean():.1f} m | "
          f"max {heights.max():.1f} m")
    print("  tallest:")
    for f in sorted(scene, key=lambda f: -f["properties"]["height"])[:6]:
        p = f["properties"]
        print(f"    {p['height']:6.1f} m  [{p['height_source']:>7}]  {p.get('name') or 'unnamed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
