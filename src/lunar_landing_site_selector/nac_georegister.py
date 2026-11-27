from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds

from lunar_landing_site_selector.micro_hazard import (
    MicroHazardResult,
    detect_craters_yolo,
    rasterize_crater_hazards,
    select_pinpoint_touchdown,
)

LUNAR_RADIUS_M = 1_737_400.0
LUNAR_GEO_CRS = CRS.from_proj4(f"+proj=longlat +R={LUNAR_RADIUS_M} +no_defs")


@dataclass
class NacCrop:
    image: np.ndarray          # (H, W) uint8 normalized for CV
    raw_band: np.ndarray       # (H, W) original raster values
    window: Window
    window_transform: Affine
    raster_crs: CRS | None
    dx_m: float
    dy_m: float
    center_lat_deg: float
    center_lon_deg: float

    @property
    def meters_per_px(self) -> float:
        return 0.5 * (self.dx_m + self.dy_m)

    def pixel_to_latlon(self, row: float, col: float) -> tuple[float, float]:
        x_crs, y_crs = self.window_transform @ (float(col) + 0.5, float(row) + 0.5)
        if self.raster_crs is None or self.raster_crs == LUNAR_GEO_CRS:
            return float(y_crs), float(x_crs)
        if self.raster_crs.is_geographic:
            lon_out, lat_out = warp_transform(self.raster_crs, LUNAR_GEO_CRS, [x_crs], [y_crs])
            return float(lat_out[0]), float(lon_out[0])
        lon_out, lat_out = warp_transform(self.raster_crs, LUNAR_GEO_CRS, [x_crs], [y_crs])
        return float(lat_out[0]), float(lon_out[0])


@dataclass
class NacTouchdownReport:
    crop: NacCrop
    boxes_xyxy: np.ndarray
    hazard: MicroHazardResult
    touchdown_lat_deg: float | None
    touchdown_lon_deg: float | None


def _normalize_to_uint8(band: np.ndarray) -> np.ndarray:
    arr = band.astype(np.float32)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid], [1.0, 99.0])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _pixel_sizes_meters(transform: Affine, crs: CRS | None, lat_deg: float) -> tuple[float, float]:
    px_w = abs(float(transform.a))
    px_h = abs(float(transform.e))
    if crs is not None and not crs.is_geographic:
        # Assume meter-based projection (e.g., Equirectangular / Polar Stereographic on Moon)
        return px_w, px_h
    # Geographic degrees -> meters on lunar sphere
    lat_rad = np.deg2rad(lat_deg)
    dy_m = np.deg2rad(px_h) * LUNAR_RADIUS_M
    dx_m = np.deg2rad(px_w) * LUNAR_RADIUS_M * max(float(np.cos(lat_rad)), 1e-6)
    return float(dx_m), float(dy_m)


def extract_nac_window(
    geotiff_path: str | Path,
    lat_deg: float,
    lon_deg: float,
    half_span_m: float = 600.0,
    band_index: int = 1,
) -> NacCrop:
    """Ambil potongan GeoTIFF LROC NAC di sekitar koordinat (lat_deg, lon_deg) Bulan."""
    with rasterio.open(geotiff_path) as src:
        raster_crs = src.crs
        src_crs = raster_crs if raster_crs is not None else LUNAR_GEO_CRS

        cx, cy = warp_transform(LUNAR_GEO_CRS, src_crs, [float(lon_deg)], [float(lat_deg)])
        center_x, center_y = float(cx[0]), float(cy[0])

        if src_crs.is_geographic:
            lat_rad = np.deg2rad(float(lat_deg))
            d_lat = np.rad2deg(half_span_m / LUNAR_RADIUS_M)
            cos_lat = max(float(np.cos(lat_rad)), 1e-6)
            d_lon = np.rad2deg(half_span_m / (LUNAR_RADIUS_M * cos_lat))
            left, right = center_x - d_lon, center_x + d_lon
            bottom, top = center_y - d_lat, center_y + d_lat
        else:
            left, right = center_x - half_span_m, center_x + half_span_m
            bottom, top = center_y - half_span_m, center_y + half_span_m

        win = from_bounds(left, bottom, right, top, transform=src.transform)
        full_win = Window(0, 0, src.width, src.height)
        try:
            clipped = win.intersection(full_win)
        except WindowError as exc:
            raise ValueError(
                f"Target ({lat_deg:.4f}, {lon_deg:.4f}) is outside GeoTIFF extent: {geotiff_path}"
            ) from exc
        if clipped.width < 1 or clipped.height < 1:
            raise ValueError(
                f"Target ({lat_deg:.4f}, {lon_deg:.4f}) is outside GeoTIFF extent: {geotiff_path}"
            )

        raw = src.read(band_index, window=clipped, boundless=False)
        win_transform = src.window_transform(clipped)

    dx_m, dy_m = _pixel_sizes_meters(win_transform, raster_crs, lat_deg)
    img_u8 = _normalize_to_uint8(raw)
    return NacCrop(
        image=img_u8,
        raw_band=raw,
        window=clipped,
        window_transform=win_transform,
        raster_crs=raster_crs,
        dx_m=dx_m,
        dy_m=dy_m,
        center_lat_deg=float(lat_deg),
        center_lon_deg=float(lon_deg),
    )


def run_nac_touchdown(
    geotiff_path: str | Path,
    lat_deg: float,
    lon_deg: float,
    weights_path: str | Path,
    half_span_m: float = 600.0,
    conf: float = 0.20,
    rim_margin_factor: float = 1.1,
    tmp_image_path: str | Path = Path("outputs/_nac_crop_tmp.png"),
) -> NacTouchdownReport:
    """Ekstrak window NAC di koordinat Fase 1, jalankan deteksi kawah, lalu petakan touchdown ke (lat, lon)."""
    import matplotlib.image as mpimg

    crop = extract_nac_window(geotiff_path, lat_deg=lat_deg, lon_deg=lon_deg, half_span_m=half_span_m)
    tmp_path = Path(tmp_image_path)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(tmp_path, crop.image, cmap="gray")

    shape, boxes, _ = detect_craters_yolo(tmp_path, weights_path, conf=conf)
    mask = rasterize_crater_hazards(shape, boxes, rim_margin_factor=rim_margin_factor)
    hazard = select_pinpoint_touchdown(mask, meters_per_px=crop.meters_per_px)

    td_lat: float | None = None
    td_lon: float | None = None
    if hazard.touchdown is not None:
        td_lat, td_lon = crop.pixel_to_latlon(hazard.touchdown.row, hazard.touchdown.col)

    return NacTouchdownReport(
        crop=crop,
        boxes_xyxy=boxes,
        hazard=hazard,
        touchdown_lat_deg=td_lat,
        touchdown_lon_deg=td_lon,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: Georeference lunar optical GeoTIFF around Phase 1 candidate.")
    parser.add_argument("geotiff", type=Path, help="Path to lunar GeoTIFF (e.g., LROC NAC).")
    parser.add_argument("--lat", type=float, required=True, help="Target latitude in degrees North.")
    parser.add_argument("--lon", type=float, required=True, help="Target longitude in degrees East.")
    parser.add_argument("--half-span-m", type=float, default=600.0, help="Half-width of crop around target in meters.")
    parser.add_argument("--weights", type=Path, default=None, help="Optional YOLO weights path for touchdown selection.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO confidence threshold.")
    args = parser.parse_args()

    if args.weights is None:
        crop = extract_nac_window(args.geotiff, lat_deg=args.lat, lon_deg=args.lon, half_span_m=args.half_span_m)
        c_lat, c_lon = crop.pixel_to_latlon(crop.image.shape[0] / 2.0, crop.image.shape[1] / 2.0)
        print(f"Crop shape       : {crop.image.shape[1]}x{crop.image.shape[0]} px")
        print(f"Pixel scale      : dx={crop.dx_m:.2f} m/px, dy={crop.dy_m:.2f} m/px")
        print(f"Crop center      : {c_lat:.6f}°N, {c_lon:.6f}°E")
    else:
        report = run_nac_touchdown(
            args.geotiff,
            lat_deg=args.lat,
            lon_deg=args.lon,
            weights_path=args.weights,
            half_span_m=args.half_span_m,
            conf=args.conf,
        )
        print(f"Crop shape       : {report.crop.image.shape[1]}x{report.crop.image.shape[0]} px ({report.crop.meters_per_px:.2f} m/px)")
        print(f"Detected craters : {len(report.boxes_xyxy)}")
        if report.hazard.touchdown is None:
            print("Touchdown        : None")
        else:
            td = report.hazard.touchdown
            print(
                f"Touchdown (r, c) : ({td.row}, {td.col}) | Clearance: {td.clearance_m:.1f} m "
                f"| ({report.touchdown_lat_deg:.6f}°N, {report.touchdown_lon_deg:.6f}°E)"
            )
