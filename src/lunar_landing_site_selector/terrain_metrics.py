from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter

from lunar_landing_site_selector.ingest import LunarDEM, load_dem_roi


@dataclass
class TerrainMetrics:
    slope_deg: np.ndarray    # 2D float32, kemiringan dalam derajat
    roughness_m: np.ndarray  # 2D float32, std dev elevasi lokal dalam meter


def compute_terrain_metrics(dem: LunarDEM, window_px: int = 5) -> TerrainMetrics:
    """Hitung peta slope (derajat) dan roughness lokal (meter) dari DEM."""
    z = dem.elevation_m.astype(np.float64)

    # Axis 0 = Y (utara-selatan), Axis 1 = X (barat-timur)
    dz_dy, dz_dx = np.gradient(z, dem.dy_m, dem.dx_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # Varians lokal cepat: E[Z^2] - E[Z]^2
    mean_z = uniform_filter(z, size=window_px, mode="nearest")
    mean_z2 = uniform_filter(z * z, size=window_px, mode="nearest")
    roughness_m = np.sqrt(np.maximum(mean_z2 - mean_z * mean_z, 0.0))

    return TerrainMetrics(
        slope_deg=slope_deg.astype(np.float32),
        roughness_m=roughness_m.astype(np.float32),
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m lunar_landing_site_selector.terrain_metrics <path_dem> [scale_to_meters]")
        sys.exit(1)

    path = Path(sys.argv[1])
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    dem = load_dem_roi(path, window=(0, 0, 2048, 2048), scale_to_meters=scale)
    metrics = compute_terrain_metrics(dem, window_px=5)

    s = metrics.slope_deg
    r = metrics.roughness_m
    print(f"Scale used: {scale}")
    print(f"Slope     : median={np.nanmedian(s):.2f}°, p90={np.nanpercentile(s, 90):.2f}°, max={np.nanmax(s):.2f}°")
    print(f"Area aman : <5° = {(s < 5).mean()*100:.1f}% | <10° = {(s < 10).mean()*100:.1f}%")
    print(f"Roughness : median={np.nanmedian(r):.2f} m, p90={np.nanpercentile(r, 90):.2f} m, max={np.nanmax(r):.2f} m")
