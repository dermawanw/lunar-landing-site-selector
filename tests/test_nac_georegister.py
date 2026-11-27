from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from lunar_landing_site_selector.nac_georegister import LUNAR_GEO_CRS, extract_nac_window


def _write_synthetic_lunar_geotiff(path: Path) -> None:
    # 200x200 pixels, 0.0005 deg/px (~15.2 m/px at ~27 deg N on Moon)
    west, north, step = 1.45, 27.12, 0.0005
    transform = from_origin(west, north, step, step)
    rng = np.random.default_rng(42)
    data = rng.integers(50, 200, size=(200, 200), dtype=np.uint8)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=LUNAR_GEO_CRS,
        transform=transform,
    ) as dst:
        dst.write(data, 1)


def test_extract_window_and_roundtrip_latlon(tmp_path):
    tif_path = tmp_path / "synthetic_nac.tif"
    _write_synthetic_lunar_geotiff(tif_path)

    target_lat, target_lon = 27.0752, 1.5049
    crop = extract_nac_window(tif_path, lat_deg=target_lat, lon_deg=target_lon, half_span_m=400.0)

    assert crop.image.ndim == 2
    assert crop.image.dtype == np.uint8
    assert crop.image.size > 0
    assert crop.dx_m > 0.0 and crop.dy_m > 0.0

    mid_r = crop.image.shape[0] / 2.0 - 0.5
    mid_c = crop.image.shape[1] / 2.0 - 0.5
    out_lat, out_lon = crop.pixel_to_latlon(mid_r, mid_c)

    assert abs(out_lat - target_lat) < 0.002
    assert abs(out_lon - target_lon) < 0.002


def test_out_of_bounds_raises_value_error(tmp_path):
    tif_path = tmp_path / "synthetic_nac.tif"
    _write_synthetic_lunar_geotiff(tif_path)

    with pytest.raises(ValueError, match="outside GeoTIFF extent"):
        extract_nac_window(tif_path, lat_deg=-10.0, lon_deg=120.0, half_span_m=400.0)


def test_projected_lunar_meter_crs_roundtrip(tmp_path):
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform

    lunar_eqc = CRS.from_proj4("+proj=eqc +R=1737400 +units=m +no_defs")
    target_lat, target_lon = 27.0752, 1.5049
    cx, cy = warp_transform(LUNAR_GEO_CRS, lunar_eqc, [target_lon], [target_lat])

    px_m = 2.0
    tif_path = tmp_path / "synthetic_nac_eqc.tif"
    transform = from_origin(cx[0] - 200.0, cy[0] + 200.0, px_m, px_m)
    data = np.full((200, 200), 128, dtype=np.uint8)

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype=data.dtype,
        crs=lunar_eqc,
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    crop = extract_nac_window(tif_path, lat_deg=target_lat, lon_deg=target_lon, half_span_m=100.0)
    assert crop.dx_m == pytest.approx(px_m)
    assert crop.dy_m == pytest.approx(px_m)
    assert crop.image.shape == (100, 100)

    mid_r = crop.image.shape[0] / 2.0 - 0.5
    mid_c = crop.image.shape[1] / 2.0 - 0.5
    out_lat, out_lon = crop.pixel_to_latlon(mid_r, mid_c)
    assert abs(out_lat - target_lat) < 1e-4
    assert abs(out_lon - target_lon) < 1e-4
