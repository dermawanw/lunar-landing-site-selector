import numpy as np

from lunar_landing_site_selector.ingest import LunarDEM
from lunar_landing_site_selector.site_selector import select_landing_sites
from lunar_landing_site_selector.terrain_metrics import TerrainMetrics, compute_terrain_metrics


def _make_dem(elevation: np.ndarray, dx_m: float = 100.0, dy_m: float = 100.0) -> LunarDEM:
    return LunarDEM(
        elevation_m=elevation.astype(np.float32),
        dx_m=dx_m,
        dy_m=dy_m,
        bounds=(0.0, 0.0, 1.0, 1.0),
    )


def test_flat_terrain_has_zero_slope_and_roughness():
    dem = _make_dem(np.full((32, 32), -2000.0))
    metrics = compute_terrain_metrics(dem, window_px=5)
    assert np.allclose(metrics.slope_deg, 0.0, atol=1e-5)
    assert np.allclose(metrics.roughness_m, 0.0, atol=1e-5)


def test_tilted_plane_matches_analytical_slope():
    xs = np.arange(32, dtype=np.float32) * 100.0
    dem = _make_dem(np.tile(xs, (32, 1)), dx_m=100.0, dy_m=100.0)
    metrics = compute_terrain_metrics(dem, window_px=5)
    assert np.allclose(metrics.slope_deg, 45.0, atol=1e-4)


def test_zone_averaged_scoring_beats_single_flat_center_pixel():
    h, w = 40, 60
    y, x = np.ogrid[:h, :w]
    island_a = np.hypot(y - 15, x - 15) <= 6
    island_b = np.hypot(y - 15, x - 45) <= 6

    slope = np.full((h, w), 25.0, dtype=np.float32)
    slope[island_a] = 4.0
    slope[15, 15] = 0.0  # cuma 1 piksel pusat yang datar
    slope[island_b] = 1.0

    rough = np.full((h, w), 1.0, dtype=np.float32)
    dem = _make_dem(np.zeros((h, w)), dx_m=100.0, dy_m=100.0)
    result = select_landing_sites(
        dem,
        TerrainMetrics(slope_deg=slope, roughness_m=rough),
        min_safe_radius_m=300.0,
        min_separation_m=1500.0,
        top_n=2,
    )

    assert (result.sites[0].row, result.sites[0].col) == (15, 45)
    assert all(s.zone_hazard_pixels == 0 for s in result.sites)


def test_nms_enforces_minimum_separation():
    dem = _make_dem(np.zeros((80, 80)), dx_m=100.0, dy_m=100.0)
    result = select_landing_sites(
        dem,
        compute_terrain_metrics(dem, window_px=5),
        min_safe_radius_m=300.0,
        min_separation_m=2000.0,
        top_n=3,
    )

    assert len(result.sites) == 3
    for i in range(len(result.sites)):
        for j in range(i + 1, len(result.sites)):
            a, b = result.sites[i], result.sites[j]
            assert np.hypot((a.row - b.row) * dem.dy_m, (a.col - b.col) * dem.dx_m) >= 2000.0
