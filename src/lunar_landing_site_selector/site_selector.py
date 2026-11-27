from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

from lunar_landing_site_selector.ingest import LunarDEM, load_dem_roi
from lunar_landing_site_selector.terrain_metrics import TerrainMetrics, compute_terrain_metrics


@dataclass
class LandingSite:
    rank: int
    row: int
    col: int
    lat_deg: float
    lon_deg: float
    score: float
    safe_radius_m: float
    zone_mean_slope_deg: float
    zone_max_slope_deg: float
    zone_mean_roughness_m: float
    zone_hazard_pixels: int


@dataclass
class SelectionResult:
    safe_mask: np.ndarray        # 2D bool: True = aman
    safe_radius_m: np.ndarray    # 2D float32: jarak ke bahaya terdekat (meter)
    score_map: np.ndarray        # 2D float32: skor [0, 1] untuk kandidat valid, NaN di luar itu
    sites: list[LandingSite]


def pixel_to_latlon(row: int, col: int, dem: LunarDEM) -> tuple[float, float]:
    """Konversi indeks (row, col) ke (lat, lon) pusat piksel."""
    h, w = dem.elevation_m.shape
    lon_min, lat_min, lon_max, lat_max = dem.bounds
    lon = lon_min + (col + 0.5) * (lon_max - lon_min) / w
    lat = lat_max - (row + 0.5) * (lat_max - lat_min) / h
    return float(lat), float(lon)


def select_landing_sites(
    dem: LunarDEM,
    metrics: TerrainMetrics,
    max_slope_deg: float = 5.0,
    max_roughness_m: float = 5.0,
    min_safe_radius_m: float = 500.0,
    min_separation_m: float = 5000.0,
    top_n: int = 3,
) -> SelectionResult:
    """Identifikasi zona aman dan pilih Top-N lokasi pendaratan terbaik."""
    safe_mask = (
        np.isfinite(dem.elevation_m)
        & (metrics.slope_deg <= max_slope_deg)
        & (metrics.roughness_m <= max_roughness_m)
    )

    # Tandai tepi ROI sebagai batas tak aman agar lingkaran aman tidak keluar peta
    bounded_mask = safe_mask.copy()
    bounded_mask[0, :] = False
    bounded_mask[-1, :] = False
    bounded_mask[:, 0] = False
    bounded_mask[:, -1] = False

    safe_radius_m = distance_transform_edt(
        bounded_mask, sampling=(dem.dy_m, dem.dx_m)
    ).astype(np.float32)

    valid = safe_radius_m >= min_safe_radius_m
    score_map = np.full(dem.elevation_m.shape, np.nan, dtype=np.float32)

    if not np.any(valid):
        return SelectionResult(safe_mask, safe_radius_m, score_map, [])

    # Normalisasi skor hanya pada kandidat yang memenuhi radius aman minimum
    r_valid = safe_radius_m[valid]
    r_span = max(float(r_valid.max() - r_valid.min()), 1e-6)
    clearance_score = (r_valid - r_valid.min()) / r_span
    slope_score = 1.0 - (metrics.slope_deg[valid] / max_slope_deg)
    rough_score = 1.0 - (metrics.roughness_m[valid] / max_roughness_m)

    score_map[valid] = 0.50 * clearance_score + 0.30 * slope_score + 0.20 * rough_score

    # Greedy Non-Maximum Suppression untuk mengambil Top-N lokasi terpisah
    h, w = dem.elevation_m.shape
    y_idx, x_idx = np.ogrid[:h, :w]
    work_scores = score_map.copy()
    sites: list[LandingSite] = []

    for rank in range(1, top_n + 1):
        if np.all(np.isnan(work_scores)):
            break

        flat_idx = int(np.nanargmax(work_scores))
        r, c = divmod(flat_idx, w)
        site_score = float(work_scores[r, c])

        dist_m = np.hypot((y_idx - r) * dem.dy_m, (x_idx - c) * dem.dx_m)
        zone = dist_m < min_safe_radius_m

        lat, lon = pixel_to_latlon(r, c, dem)
        sites.append(
            LandingSite(
                rank=rank,
                row=r,
                col=c,
                lat_deg=lat,
                lon_deg=lon,
                score=site_score,
                safe_radius_m=float(safe_radius_m[r, c]),
                zone_mean_slope_deg=float(metrics.slope_deg[zone].mean()),
                zone_max_slope_deg=float(metrics.slope_deg[zone].max()),
                zone_mean_roughness_m=float(metrics.roughness_m[zone].mean()),
                zone_hazard_pixels=int((~safe_mask[zone]).sum()),
            )
        )

        work_scores[dist_m < min_separation_m] = np.nan

    return SelectionResult(safe_mask, safe_radius_m, score_map, sites)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m lunar_landing_site_selector.site_selector <path_dem> [scale_to_meters]")
        sys.exit(1)

    path = Path(sys.argv[1])
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    dem = load_dem_roi(path, window=(0, 0, 2048, 2048), scale_to_meters=scale)
    metrics = compute_terrain_metrics(dem, window_px=5)
    result = select_landing_sites(dem, metrics)

    print(f"Safe area     : {result.safe_mask.mean() * 100:.1f}%")
    print(f"Max clearance : {result.safe_radius_m.max():.0f} m\n")

    if not result.sites:
        print("Tidak ada lokasi yang memenuhi syarat. Coba longgarkan threshold.")
    else:
        header = f"{'Rank':<5} {'Lat (°N)':<10} {'Lon (°E)':<10} {'Score':<7} {'Clearance':<11} {'Mean/Max Slope':<16} {'Mean Rough':<11} {'Hazards'}"
        print(header)
        print("-" * len(header))
        for s in result.sites:
            print(
                f"#{s.rank:<4} {s.lat_deg:<10.4f} {s.lon_deg:<10.4f} {s.score:<7.3f} "
                f"{s.safe_radius_m:<9.0f} m {s.zone_mean_slope_deg:>5.2f}° / {s.zone_max_slope_deg:<5.2f}°  "
                f"{s.zone_mean_roughness_m:<8.2f} m  {s.zone_hazard_pixels}"
            )
