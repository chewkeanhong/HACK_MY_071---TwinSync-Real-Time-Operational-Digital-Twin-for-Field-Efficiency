"""Monsoon weather: the field, the rain fade, and the flood-routing fusion."""

from __future__ import annotations

import numpy as np
import pytest

from twinsync.routing import RoadNetwork
from twinsync.terrain import Terrain
from twinsync.weather import (
    BACKHAUL_GHZ,
    FLOOD_RAIN_MM_HR,
    LINK_MARGIN_DB,
    StormCell,
    WeatherField,
    fade_to_capacity,
    rain_fade_db,
)


def cell(**overrides) -> StormCell:
    spec = dict(x=0.0, y=0.0, radius_m=1000.0, peak_mm_hr=80.0,
                start_s=0.0, duration_s=1000.0,
                drift_bearing_deg=90.0, drift_kmh=36.0)
    spec.update(overrides)
    return StormCell(**spec)


# -- the cell ----------------------------------------------------------


def test_cell_drifts_at_the_stated_speed():
    """36 km/h due east for 100 s is 1000 m east."""
    moving = cell()
    x0, y0 = moving.centre_at(0.0)
    x1, y1 = moving.centre_at(100.0)
    assert x1 - x0 == pytest.approx(1000.0, rel=1e-6)
    assert y1 - y0 == pytest.approx(0.0, abs=1e-6)


def test_bearing_is_compass_not_maths():
    """0 degrees must be north, not east -- an easy and silent thing to get wrong."""
    north = cell(drift_bearing_deg=0.0).centre_at(100.0)
    assert north[1] > 900.0 and abs(north[0]) < 1e-6


def test_envelope_is_zero_outside_the_cell_lifetime():
    c = cell(start_s=100.0, duration_s=200.0)
    assert c.envelope(99.0) == 0.0
    assert c.envelope(301.0) == 0.0
    assert c.envelope(200.0) == pytest.approx(1.0, abs=1e-9)


def test_rain_is_strongest_at_the_centre_and_falls_off():
    c = cell(drift_kmh=0.0)
    mid = c.start_s + c.duration_s / 2.0
    centre = float(np.atleast_1d(c.rain_at(0.0, 0.0, mid))[0])
    edge = float(np.atleast_1d(c.rain_at(1000.0, 0.0, mid))[0])
    far = float(np.atleast_1d(c.rain_at(5000.0, 0.0, mid))[0])
    assert centre > edge > far
    assert centre == pytest.approx(80.0, rel=1e-6)
    assert far < 0.01


def test_field_superposes_cells():
    field = WeatherField([cell(drift_kmh=0.0), cell(drift_kmh=0.0)])
    single = WeatherField([cell(drift_kmh=0.0)])
    mid = 500.0
    assert (field.at(0.0, 0.0, mid)["rainfall_mm_hr"]
            == pytest.approx(2 * single.at(0.0, 0.0, mid)["rainfall_mm_hr"], rel=1e-6))


def test_field_is_deterministic():
    a = WeatherField([cell()]).at(100.0, 100.0, 400.0)
    b = WeatherField([cell()]).at(100.0, 100.0, 400.0)
    assert a == b


def test_clear_sky_when_no_cells():
    field = WeatherField([])
    conditions = field.at(0.0, 0.0, 500.0)
    assert conditions["rainfall_mm_hr"] == 0.0
    assert conditions["lightning_per_km2_hr"] == 0.0
    assert field.any_cells is False


def test_humidity_and_wind_rise_with_rain():
    field = WeatherField([cell(drift_kmh=0.0)], baseline={"humidity_pct": 80.0,
                                                          "wind_kmh": 10.0})
    dry = field.at(6000.0, 0.0, 500.0)
    wet = field.at(0.0, 0.0, 500.0)
    assert wet["humidity_pct"] > dry["humidity_pct"]
    assert wet["wind_kmh"] > dry["wind_kmh"]
    assert wet["humidity_pct"] <= 100.0


# -- rain fade ---------------------------------------------------------


def test_rain_fade_increases_monotonically_with_rate():
    fades = [rain_fade_db(rate, 2.0) for rate in (0.0, 10.0, 40.0, 100.0)]
    assert fades == sorted(fades)
    assert fades[0] == 0.0


def test_rain_fade_scales_with_path_length():
    assert rain_fade_db(50.0, 4.0) == pytest.approx(2 * rain_fade_db(50.0, 2.0), rel=1e-9)


def test_rain_fade_averages_along_the_path():
    """A hop half in the clear must not be attenuated as if it were all in the rain."""
    uniform = rain_fade_db([80.0, 80.0, 80.0, 80.0], 2.0)
    partial = rain_fade_db([80.0, 80.0, 0.0, 0.0], 2.0)
    assert partial == pytest.approx(uniform / 2.0, rel=1e-9)


def test_fade_to_capacity_degrades_then_fails():
    assert fade_to_capacity(0.0) == pytest.approx(1.0, rel=1e-9)
    assert 0.0 < fade_to_capacity(20.0) < 1.0
    assert fade_to_capacity(LINK_MARGIN_DB) == 0.0
    assert fade_to_capacity(LINK_MARGIN_DB + 10.0) == 0.0


def test_capacity_is_monotonic_in_fade():
    values = [fade_to_capacity(f) for f in range(0, 40, 4)]
    assert values == sorted(values, reverse=True)


def test_backhaul_band_is_microwave_not_access():
    """Rain fade belongs on the backhaul. At 2.1 GHz it would be a fabricated effect."""
    assert BACKHAUL_GHZ >= 10.0


def test_backhaul_capacity_drops_under_a_cell():
    field = WeatherField([cell(peak_mm_hr=120.0, radius_m=2000.0, drift_kmh=0.0)])
    a, b = np.array([-1500.0, 0.0]), np.array([1500.0, 0.0])

    clear, clear_fade = field.backhaul_capacity(a, b, 0.0)
    wet, wet_fade = field.backhaul_capacity(a, b, 500.0)

    assert clear == pytest.approx(1.0, rel=1e-9) and clear_fade == 0.0
    assert wet < clear
    assert wet_fade > 0.0


# -- flooding: DEM + rain + road graph ---------------------------------


def build_terrain() -> Terrain:
    """A bowl: low in the middle, high at the edges."""
    size = 40
    ys, xs = np.mgrid[0:size, 0:size]
    grid = 20.0 + 0.6 * ((xs - size / 2) ** 2 + (ys - size / 2) ** 2) ** 0.5
    from twinsync.terrain import TerrainMeta
    meta = TerrainMeta("test", "", False, 50.0, float(grid.min()), float(grid.max()))
    return Terrain(grid, -1000.0, -1000.0, 50.0, meta)


def build_network(tmp_path) -> RoadNetwork:
    import json

    from twinsync.geo import LocalFrame
    frame = LocalFrame(101.71, 3.15)

    def line(x0, y0, x1, y1, name):
        a = frame.to_lonlat(x0, y0)
        b = frame.to_lonlat(x1, y1)
        return {"type": "Feature",
                "properties": {"highway": "secondary", "name": name, "oneway": "no"},
                "geometry": {"type": "LineString",
                             "coordinates": [[float(a[0]), float(a[1])],
                                             [float(b[0]), float(b[1])]]}}

    data = {"type": "FeatureCollection", "features": [
        line(-100.0, 0.0, 100.0, 0.0, "valley road"),      # through the low centre
        line(-900.0, 900.0, -700.0, 900.0, "ridge road"),  # up on the rim
    ]}
    path = tmp_path / "roads.geojson"
    path.write_text(json.dumps(data), encoding="utf-8")
    return RoadNetwork.load(path, frame)


def test_flooding_hits_low_ground_only(tmp_path):
    """The fusion claim: DEM says where water pools, weather says where it is falling."""
    terrain = build_terrain()
    network = build_network(tmp_path)
    field = WeatherField([cell(peak_mm_hr=140.0, radius_m=4000.0, drift_kmh=0.0)])

    affected = field.flooded_segments(network, terrain, 500.0)
    assert affected > 0

    flooded = {data.get("name") for _, _, data in network.graph.edges(data=True)
               if data.get("flooded")}
    assert "valley road" in flooded
    assert "ridge road" not in flooded, "high ground must not flood"


def test_flooding_recedes_and_restores_scenario_congestion(tmp_path):
    """Lifting the flood must not erase a standing rush-hour jam underneath it."""
    terrain = build_terrain()
    network = build_network(tmp_path)
    network.set_congestion(4.0, highway="secondary")
    field = WeatherField([cell(peak_mm_hr=140.0, radius_m=4000.0, drift_kmh=0.0,
                               start_s=0.0, duration_s=1000.0)])

    field.flooded_segments(network, terrain, 500.0)
    field.flooded_segments(network, terrain, 2000.0)      # cell has expired

    for _, _, data in network.graph.edges(data=True):
        # Edges that never flooded carry no flag at all, which is fine.
        assert data.get("flooded", False) is False
        assert data["congestion"] == pytest.approx(4.0)
        assert data["time"] == pytest.approx(data["base_time"] * 4.0)


def test_light_rain_does_not_flood(tmp_path):
    terrain = build_terrain()
    network = build_network(tmp_path)
    drizzle = WeatherField([cell(peak_mm_hr=FLOOD_RAIN_MM_HR / 4.0,
                                 radius_m=4000.0, drift_kmh=0.0)])
    assert drizzle.flooded_segments(network, terrain, 500.0) == 0


def test_active_cells_reports_only_live_ones():
    field = WeatherField([cell(start_s=100.0, duration_s=200.0)])
    assert field.active_cells(50.0) == []
    live = field.active_cells(200.0)
    assert len(live) == 1
    assert live[0]["intensity"] == pytest.approx(1.0, abs=1e-6)
    assert live[0]["rain_mm_hr"] > 0.0
