"""The pipeline must be coordinate-driven, not tuned to Kuala Lumpur.

This is an ASEAN competition and the demo is one Malaysian AOI, so "does this transfer?"
is a fair question. The honest answer is that the *code* transfers and only the bake is
KL-specific -- but that is a claim, and a claim in a README is worth less than a test.

These check the two places a hardcoded AOI would hide: the Copernicus tile naming, which
has to pick the right 1x1 degree cell anywhere in the region, and the runtime modules,
which must contain no coordinates at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Capital-city coordinates across the region, with the Copernicus GLO-30 tile that
# contains each. Tile naming is {N|S}{lat:02d}_00_{E|W}{lon:03d}_00 over the floor of
# the coordinate, so these double as a floor-toward-negative-infinity check south of
# the equator -- where the naive int() truncation gives the wrong tile.
ASEAN_CITIES = [
    ("Kuala Lumpur", 101.7132, 3.1497, "N03_00_E101_00"),
    ("Singapore", 103.8198, 1.3521, "N01_00_E103_00"),
    ("Bangkok", 100.5018, 13.7563, "N13_00_E100_00"),
    ("Jakarta", 106.8456, -6.2088, "S07_00_E106_00"),
    ("Manila", 120.9842, 14.5995, "N14_00_E120_00"),
    ("Hanoi", 105.8342, 21.0278, "N21_00_E105_00"),
    ("Phnom Penh", 104.9282, 11.5564, "N11_00_E104_00"),
]


def load_tile_name():
    """Import the DEM fetcher's tile naming without importing rasterio."""
    import importlib.util
    import sys

    path = ROOT / "scripts" / "fetch_dem.py"
    spec = importlib.util.spec_from_file_location("_fetch_dem", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_fetch_dem"] = module
    spec.loader.exec_module(module)
    return module.tile_name


tile_name = load_tile_name()


@pytest.mark.parametrize("city,lon,lat,expected", ASEAN_CITIES,
                         ids=[c[0] for c in ASEAN_CITIES])
def test_dem_tile_resolves_across_the_region(city, lon, lat, expected):
    assert tile_name(lon, lat) == f"Copernicus_DSM_COG_10_{expected}_DEM"


def test_southern_hemisphere_floors_rather_than_truncates():
    """Jakarta is at -6.21, which lives in the S07 tile, not S06.

    `int(-6.21)` is -6 and would fetch the wrong tile -- a silent 100 km offset that
    would put every building on the wrong ground.
    """
    assert "S07" in tile_name(106.8456, -6.2088)


# -- no coordinates baked into the runtime ------------------------------

RUNTIME_DIRS = ("twinsync", "edge")

# Longitude only. A latitude band of 3.0-4.0 would match any ordinary constant -- an
# earlier version of this test flagged a 3.6 km/h conversion and a 3.2 second timeout --
# whereas a bare 101.x in runtime code is a Kuala Lumpur coordinate and nothing else.
KL_LON = (101.0, 102.0)


def literals(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            yield node.value


def test_runtime_modules_contain_no_aoi_coordinates():
    """A default lon/lat in the twin itself would make a second city silently wrong.

    Scripts are exempt: their argparse defaults are meant to name the AOI being baked.
    """
    offenders = []
    for directory in RUNTIME_DIRS:
        for path in sorted((ROOT / directory).glob("*.py")):
            for value in literals(path):
                if KL_LON[0] < value < KL_LON[1]:
                    offenders.append(f"{path.relative_to(ROOT)}: {value}")
    assert not offenders, ("AOI coordinates baked into runtime code: "
                           + ", ".join(offenders))


def test_server_honours_a_data_directory_override():
    """Running a second AOI is an env var, not a code change."""
    source = (ROOT / "twinsync" / "server.py").read_text(encoding="utf-8")
    assert 'os.environ.get("TWINSYNC_DATA"' in source


def test_the_fetchers_take_the_aoi_as_an_argument():
    osm = (ROOT / "scripts" / "fetch_osm.py").read_text(encoding="utf-8")
    ndvi = (ROOT / "scripts" / "fetch_ndvi.py").read_text(encoding="utf-8")
    assert '"--bbox"' in osm, "fetch_osm must accept an AOI bbox"
    # fetch_ndvi derives its search box from the towers it is given, so it needs no
    # bbox flag -- but it must not carry a hardcoded one either.
    assert "101.7" not in ndvi
