from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.windows import Window

MOON_RADIUS_M = 1_737_400.0
DEG_TO_M = (np.pi / 180.0) * MOON_RADIUS_M


@dataclass
class LunarDEM:
    elevation_m: np.ndarray  # 2D float32, dalam meter (NaN untuk nodata)
    dx_m: float              # resolusi horizontal (meter/piksel)
    dy_m: float              # resolusi vertikal (meter/piksel)
    bounds: tuple[float, float, float, float]  # (left, bottom, right, top)


def load_dem_roi(
    filepath: str | Path,
    window: tuple[int, int, int, int] | None = (0, 0, 2048, 2048),
    scale_to_meters: float = 1.0,
) -> LunarDEM:
    """Muat potongan ROI dari DEM Bulan dan kembalikan elevasi dalam meter.

    Args:
        filepath: Path ke file .JP2, .TIF, atau .LBL (untuk pasangan FLOAT_IMG).
        window: (col_off, row_off, width, height) atau None untuk seluruh tile.
        scale_to_meters: Pengali manual ke meter (misal 1000.0 jika data dalam km,
            0.5 jika JP2 16-bit berskala 0.5 m dan GDAL tidak auto-scale).
    """
    win = Window(*window) if window is not None else None

    with rasterio.open(filepath) as src:
        raw = src.read(1, window=win).astype(np.float32)

        if src.nodata is not None:
            raw[raw == src.nodata] = np.nan

        # Terapkan metadata scale/offset GDAL jika ada, lalu pengali manual
        gdal_scale = src.scales[0] if src.scales else 1.0
        gdal_offset = src.offsets[0] if src.offsets else 0.0
        elevation_m = (raw * gdal_scale + gdal_offset) * scale_to_meters

        bounds = src.window_bounds(win) if win is not None else tuple(src.bounds)
        res_x, res_y = src.res

        # Deteksi satuan piksel: jika < 1.0, koordinat masih dalam derajat
        if res_x < 1.0 and res_y < 1.0:
            center_lat_deg = 0.5 * (bounds[1] + bounds[3])
            dy_m = res_y * DEG_TO_M
            dx_m = res_x * DEG_TO_M * np.cos(np.radians(center_lat_deg))
        else:
            dx_m, dy_m = float(res_x), float(res_y)

    # Sanity check: rentang elevasi ROI Bulan hampir tidak mungkin < 50 m
    span = float(np.nanmax(elevation_m) - np.nanmin(elevation_m))
    if span < 50.0:
        print(
            f"[PERINGATAN] Rentang elevasi hanya {span:.2f}. "
            "Kemungkinan data masih dalam kilometer—coba scale_to_meters=1000.0."
        )

    return LunarDEM(elevation_m=elevation_m, dx_m=dx_m, dy_m=dy_m, bounds=bounds)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python -m lunar_landing_site_selector.ingest <path_dem> [scale_to_meters]")
        sys.exit(1)

    path = Path(sys.argv[1])
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    dem = load_dem_roi(path, window=(0, 0, 2048, 2048), scale_to_meters=scale)

    nan_pct = 100.0 * np.isnan(dem.elevation_m).mean()
    print(f"Shape     : {dem.elevation_m.shape}")
    print(f"Pixel size: dx={dem.dx_m:.2f} m, dy={dem.dy_m:.2f} m")
    print(f"Elevation : min={np.nanmin(dem.elevation_m):.1f} m, max={np.nanmax(dem.elevation_m):.1f} m")
    print(f"NaN ratio : {nan_pct:.2f}%")