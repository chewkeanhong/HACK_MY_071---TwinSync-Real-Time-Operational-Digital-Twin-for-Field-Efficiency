"""Ground elevation, read from the baked Copernicus DEM grid.

Runtime half of the DEM story. `scripts/fetch_dem.py` does the expensive part once --
reads the GLO-30 tile off the AWS open-data bucket, approximates a DTM, resamples onto
a regular grid in local metres -- and commits the result as `data/terrain.json`. This
module reads that file with nothing but numpy, so the demo needs no rasterio, no GDAL
and no network.

Three consumers, three different questions:

* :meth:`Terrain.elevation_at` -- what is the ground height under this point? Used to
  put building bases and tower feet at their real altitude instead of a flat z=0.
* :meth:`Terrain.profile` -- what does the ground do *between* these two points? Used
  by the coverage engine to test whether a hillside blocks a radio path, which a
  building-only obstruction model cannot see.
* :meth:`Terrain.is_low_lying` -- is this point in a hollow? Used to decide which road
  segments flood when the monsoon cell parks over them.

Everything is in the same local-metre frame as the rest of the twin
(:class:`twinsync.geo.LocalFrame`), so callers pass the same `xy` they already have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Points at or below this percentile of the AOI's elevation distribution count as
# low-lying. A tenth is deliberately conservative: it picks out the genuine drainage
# lines rather than half the map, so "flooded" stays a meaningful label.
LOW_LYING_PERCENTILE = 10.0

# Samples along a path when nothing more specific is asked for. At 30 m grid spacing a
# 650 m tower range is ~22 cells, so 48 samples oversamples the grid about 2x -- enough
# that a narrow ridge cannot slip between two samples.
DEFAULT_PROFILE_SAMPLES = 48


@dataclass(frozen=True)
class TerrainMeta:
    """Provenance, carried so the UI and /api/models can be honest about the source."""

    source: str                 # "copernicus-glo30" or "synthetic"
    tile: str
    smoothed: bool
    cell_m: float
    min_elev: float
    max_elev: float

    @property
    def is_real(self) -> bool:
        return self.source.startswith("copernicus")

    def describe(self) -> str:
        kind = "Copernicus DEM GLO-30" if self.is_real else "synthetic (no DEM tile)"
        note = ", DSM opened toward DTM" if self.smoothed else ""
        return (f"{kind}{note} | {self.cell_m:.0f} m grid | "
                f"{self.min_elev:.0f}-{self.max_elev:.0f} m")


class Terrain:
    """A regular elevation grid in local metres, with bilinear sampling."""

    def __init__(self, grid: np.ndarray, min_x: float, min_y: float, cell_m: float,
                 meta: TerrainMeta):
        self.grid = np.asarray(grid, dtype=np.float64)
        self.ny, self.nx = self.grid.shape
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.cell_m = float(cell_m)
        self.meta = meta

        # Gradient in metres of rise per metre of run, precomputed once. np.gradient
        # returns d/drow, d/dcol, and rows advance north, so dz_dy comes first.
        dz_dy, dz_dx = np.gradient(self.grid, self.cell_m)
        self._slope_grid = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

        self._low_lying_threshold = float(
            np.percentile(self.grid, LOW_LYING_PERCENTILE)
        )

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Terrain":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        nx, ny = int(payload["nx"]), int(payload["ny"])
        grid = np.asarray(payload["elevations"], dtype=np.float64).reshape(ny, nx)
        meta = TerrainMeta(
            source=payload.get("source", "unknown"),
            tile=payload.get("tile", ""),
            smoothed=bool(payload.get("smoothed_dsm_to_dtm", False)),
            cell_m=float(payload["cell_m"]),
            min_elev=float(payload.get("min_elev", grid.min())),
            max_elev=float(payload.get("max_elev", grid.max())),
        )
        return cls(grid, payload["min_x"], payload["min_y"], payload["cell_m"], meta)

    @classmethod
    def flat(cls, elevation: float = 0.0) -> "Terrain":
        """A featureless surface, for tests and for running without a DEM present.

        Every terrain-aware code path stays live and simply concludes nothing is in the
        way, which is exactly the pre-DEM behaviour of the twin.
        """
        meta = TerrainMeta("none", "", False, 30.0, elevation, elevation)
        return cls(np.full((2, 2), float(elevation)), -1e6, -1e6, 2e6, meta)

    # -- sampling --------------------------------------------------------

    def _fractional(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        fx = (np.asarray(x, dtype=np.float64) - self.min_x) / self.cell_m
        fy = (np.asarray(y, dtype=np.float64) - self.min_y) / self.cell_m
        # Clamping rather than extrapolating: a ray leaving the AOI should meet the
        # edge elevation, not a linear ramp off to nonsense.
        return (np.clip(fx, 0.0, self.nx - 1.0000001),
                np.clip(fy, 0.0, self.ny - 1.0000001))

    def _sample(self, source: np.ndarray, x, y) -> np.ndarray:
        fx, fy = self._fractional(x, y)
        x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - x0, fy - y0
        x1 = np.minimum(x0 + 1, self.nx - 1)
        y1 = np.minimum(y0 + 1, self.ny - 1)
        return ((1 - tx) * (1 - ty) * source[y0, x0]
                + tx * (1 - ty) * source[y0, x1]
                + (1 - tx) * ty * source[y1, x0]
                + tx * ty * source[y1, x1])

    def elevation_at(self, x, y) -> np.ndarray | float:
        """Ground height in metres above sea level. Scalar in, scalar out."""
        result = self._sample(self.grid, x, y)
        return float(result) if np.ndim(result) == 0 else result

    def slope_at(self, x, y) -> np.ndarray | float:
        """Terrain slope in degrees. A LightGBM risk feature and a siting proxy."""
        result = self._sample(self._slope_grid, x, y)
        return float(result) if np.ndim(result) == 0 else result

    def profile(self, a_xy: np.ndarray, b_xy: np.ndarray,
                samples: int = DEFAULT_PROFILE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
        """Ground elevation along the straight line a -> b.

        Returns ``(t, elevation)`` where ``t`` runs 0..1 along the path, so callers can
        line it up against a radio ray parameterised the same way.
        """
        a = np.asarray(a_xy, dtype=np.float64)
        b = np.asarray(b_xy, dtype=np.float64)
        t = np.linspace(0.0, 1.0, max(2, int(samples)))
        xs = a[0] + (b[0] - a[0]) * t
        ys = a[1] + (b[1] - a[1]) * t
        return t, self._sample(self.grid, xs, ys)

    def is_low_lying(self, x, y) -> np.ndarray | bool:
        """True where the ground sits in the bottom decile of the AOI.

        This is the flood-proneness proxy. Real flood modelling wants flow accumulation
        and drainage capacity; elevation percentile is the honest cheap stand-in, and
        it is labelled as such wherever it surfaces.
        """
        result = np.asarray(self.elevation_at(x, y)) <= self._low_lying_threshold
        return bool(result) if np.ndim(result) == 0 else result

    @property
    def low_lying_threshold(self) -> float:
        return self._low_lying_threshold

    # -- reporting -------------------------------------------------------

    def fingerprint(self) -> str:
        """Cheap content hash, mixed into the coverage cache key.

        Without this a cached coverage result computed against flat ground would
        silently survive a terrain refresh and quietly invalidate every LOS answer.
        """
        from hashlib import sha256
        digest = sha256()
        digest.update(self.meta.source.encode("utf-8"))
        digest.update(np.round(self.grid, 2).tobytes())
        return digest.hexdigest()[:16]

    def summary(self) -> str:
        return (f"{self.nx}x{self.ny} @ {self.cell_m:.0f} m | {self.meta.describe()} | "
                f"low-lying below {self._low_lying_threshold:.1f} m")
