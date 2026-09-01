"""The DEM: sampling, slope, profiles and the flat fallback."""

from __future__ import annotations

import json

import numpy as np
import pytest

from twinsync.terrain import Terrain, TerrainMeta


def ramp_terrain(cell_m: float = 10.0) -> Terrain:
    """Ground rising 1 m per metre east, flat north-south. Slope is exactly 45 deg."""
    grid = np.tile(np.arange(10, dtype=np.float64) * cell_m, (10, 1))
    meta = TerrainMeta("test", "", False, cell_m, float(grid.min()), float(grid.max()))
    return Terrain(grid, 0.0, 0.0, cell_m, meta)


def test_grid_corners_sample_exactly():
    terrain = ramp_terrain()
    assert terrain.elevation_at(0.0, 0.0) == pytest.approx(0.0)
    assert terrain.elevation_at(10.0, 0.0) == pytest.approx(10.0)
    assert terrain.elevation_at(30.0, 50.0) == pytest.approx(30.0)


def test_bilinear_interpolation_between_corners():
    terrain = ramp_terrain()
    assert terrain.elevation_at(5.0, 0.0) == pytest.approx(5.0)
    assert terrain.elevation_at(15.0, 25.0) == pytest.approx(15.0)


def test_sampling_outside_the_grid_clamps_to_the_edge():
    """A ray leaving the AOI should meet edge elevation, not a linear ramp to nonsense."""
    terrain = ramp_terrain()
    assert terrain.elevation_at(-5000.0, 0.0) == pytest.approx(0.0)
    assert terrain.elevation_at(50000.0, 0.0) == pytest.approx(90.0)


def test_slope_of_a_known_ramp():
    """1 m rise per 1 m run is 45 degrees. Gets the gradient axis order right or not."""
    terrain = ramp_terrain()
    assert terrain.slope_at(45.0, 45.0) == pytest.approx(45.0, abs=0.5)


def test_flat_ground_has_zero_slope():
    grid = np.full((8, 8), 30.0)
    meta = TerrainMeta("test", "", False, 30.0, 30.0, 30.0)
    terrain = Terrain(grid, 0.0, 0.0, 30.0, meta)
    assert terrain.slope_at(100.0, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_profile_runs_end_to_end():
    terrain = ramp_terrain()
    t, ground = terrain.profile(np.array([0.0, 0.0]), np.array([90.0, 0.0]), 10)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert ground[0] == pytest.approx(0.0)
    assert ground[-1] == pytest.approx(90.0)
    assert np.all(np.diff(ground) > 0), "a monotonic ramp must profile monotonically"


def test_low_lying_picks_the_bottom_decile():
    terrain = ramp_terrain()
    assert terrain.is_low_lying(0.0, 0.0) is True
    assert terrain.is_low_lying(90.0, 0.0) is False


def test_flat_terrain_reproduces_pre_dem_behaviour():
    """Terrain.flat() must keep every terrain-aware path live but inert."""
    terrain = Terrain.flat()
    assert terrain.elevation_at(1234.0, -987.0) == 0.0
    assert terrain.slope_at(1234.0, -987.0) == pytest.approx(0.0)
    assert terrain.meta.source == "none"


def test_fingerprint_changes_with_the_surface():
    """The coverage cache keys on this; a collision would silently serve stale LOS."""
    a = ramp_terrain()
    b = ramp_terrain(cell_m=20.0)
    assert a.fingerprint() != b.fingerprint()
    assert a.fingerprint() == ramp_terrain().fingerprint()


def test_round_trip_through_json(tmp_path):
    terrain = ramp_terrain()
    payload = {
        "source": "copernicus-glo30", "tile": "T", "smoothed_dsm_to_dtm": True,
        "min_x": 0.0, "min_y": 0.0, "nx": 10, "ny": 10, "cell_m": 10.0,
        "min_elev": 0.0, "max_elev": 90.0,
        "elevations": [round(float(v), 2) for v in terrain.grid.ravel()],
    }
    path = tmp_path / "terrain.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = Terrain.load(path)
    assert loaded.nx == 10 and loaded.ny == 10
    assert loaded.meta.is_real is True
    assert loaded.elevation_at(30.0, 30.0) == pytest.approx(30.0)


def test_committed_dem_is_real_and_plausible():
    """Guards against a synthetic surface being committed by accident."""
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "data" / "terrain.json"
    if not path.exists():
        pytest.skip("no terrain baked -- run scripts/fetch_dem.py")

    terrain = Terrain.load(path)
    assert terrain.meta.source == "copernicus-glo30", "committed DEM must be real data"
    # The Klang valley floor sits around 20-70 m. A grid outside that has gone wrong.
    assert 5.0 < terrain.meta.min_elev < 60.0
    assert 30.0 < terrain.meta.max_elev < 200.0
