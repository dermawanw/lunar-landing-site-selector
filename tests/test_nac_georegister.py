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


def test_run_nac_touchdown_writes_plot_and_maps_latlon(tmp_path, monkeypatch):
    import matplotlib.image as mpimg
    import lunar_landing_site_selector.nac_georegister as nac_mod

    tif_path = tmp_path / "synthetic_nac.tif"
    _write_synthetic_lunar_geotiff(tif_path)

    def _fake_yolo(image_path, weights_path, conf=0.20):
        h, w = mpimg.imread(str(image_path)).shape[:2]
        boxes = np.array([[5.0, 5.0, 15.0, 15.0]], dtype=np.float32)
        return (h, w), boxes, np.array([0.9], dtype=np.float32)

    monkeypatch.setattr(nac_mod, "detect_craters_yolo", _fake_yolo)
    plot_path = tmp_path / "touchdown_plot.png"
    report = nac_mod.run_nac_touchdown(
        tif_path,
        lat_deg=27.0752,
        lon_deg=1.5049,
        weights_path=tmp_path / "dummy.pt",
        half_span_m=400.0,
        tmp_image_path=tmp_path / "tmp_crop.png",
        output_plot_path=plot_path,
    )

    assert plot_path.exists() and plot_path.stat().st_size > 0
    assert report.hazard.touchdown is not None
    assert report.touchdown_lat_deg is not None
    assert report.touchdown_lon_deg is not None


def test_dem_slope_fusion_blocks_steep_region_and_shifts_touchdown(tmp_path, monkeypatch):
    from rasterio.crs import CRS
    from rasterio.warp import transform as warp_transform
    import lunar_landing_site_selector.nac_georegister as nac_mod

    lunar_eqc = CRS.from_proj4("+proj=eqc +R=1737400 +units=m +no_defs")
    target_lat, target_lon = 27.0752, 1.5049
    cx, cy = warp_transform(LUNAR_GEO_CRS, lunar_eqc, [target_lon], [target_lat])

    # NAC 100x100 @ 2 m/px
    nac_tif = tmp_path / "nac_eqc.tif"
    with rasterio.open(
        nac_tif, "w", driver="GTiff", height=100, width=100, count=1,
        dtype=np.uint8, crs=lunar_eqc, transform=from_origin(cx[0] - 100.0, cy[0] + 100.0, 2.0, 2.0),
    ) as dst:
        dst.write(np.full((100, 100), 120, dtype=np.uint8), 1)

    # DEM 50x50 @ 4 m/px (2x coarser): left half tilted 15 deg, right half flat 0 deg
    dem_tif = tmp_path / "dem_eqc.tif"
    cols = np.arange(50, dtype=np.float32)
    profile = np.minimum(cols, 25.0) * 4.0 * np.tan(np.deg2rad(15.0))
    dem_data = np.broadcast_to(profile[None, :], (50, 50)).astype(np.float32)
    with rasterio.open(
        dem_tif, "w", driver="GTiff", height=50, width=50, count=1,
        dtype=np.float32, crs=lunar_eqc, transform=from_origin(cx[0] - 100.0, cy[0] + 100.0, 4.0, 4.0),
    ) as dst:
        dst.write(dem_data, 1)

    def _no_craters(image_path, weights_path, conf=0.20):
        return (100, 100), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    monkeypatch.setattr(nac_mod, "detect_craters_yolo", _no_craters)
    report = nac_mod.run_nac_touchdown(
        nac_tif,
        lat_deg=target_lat,
        lon_deg=target_lon,
        weights_path=tmp_path / "dummy.pt",
        half_span_m=100.0,
        tmp_image_path=tmp_path / "tmp.png",
        dem_path=dem_tif,
        max_slope_deg=5.0,
    )

    assert report.hazard.hazard_mask[:, :45].all()
    assert not report.hazard.hazard_mask[:, 55:].any()
    assert report.hazard.touchdown is not None
    assert report.hazard.touchdown.col > 55


def test_dem_without_overlap_raises_value_error(tmp_path):
    from lunar_landing_site_selector.nac_georegister import compute_dem_slope_hazard

    nac_tif = tmp_path / "nac.tif"
    _write_synthetic_lunar_geotiff(nac_tif)
    crop = extract_nac_window(nac_tif, lat_deg=27.0752, lon_deg=1.5049, half_span_m=400.0)

    # DEM di belahan lain Bulan
    far_dem = tmp_path / "far_dem.tif"
    with rasterio.open(
        far_dem, "w", driver="GTiff", height=20, width=20, count=1,
        dtype=np.float32, crs=LUNAR_GEO_CRS, transform=from_origin(120.0, -10.0, 0.001, 0.001),
    ) as dst:
        dst.write(np.zeros((20, 20), dtype=np.float32), 1)

    with pytest.raises(ValueError, match="does not overlap"):
        compute_dem_slope_hazard(far_dem, crop, max_slope_deg=5.0)


def _load_main_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lunar_main_cli",
        Path(__file__).resolve().parents[1] / "main.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_main_cli_nac_dem_requires_geotiff_and_weights():
    main_mod = _load_main_module()
    with pytest.raises(SystemExit):
        main_mod.parse_args(["--nac-dem", "dem.tif"])
    with pytest.raises(SystemExit):
        main_mod.parse_args(["--nac-geotiff", "nac.tif", "--nac-dem", "dem.tif"])


def test_main_forwards_nac_dem_and_plot_path(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lunar_landing_site_selector.nac_georegister as nac_mod

    main_mod = _load_main_module()
    nac_tif = tmp_path / "nac.tif"
    weights = tmp_path / "best.pt"
    nac_dem = tmp_path / "dem.tif"
    for p in (nac_tif, weights, nac_dem):
        p.write_bytes(b"stub")

    fake_site = SimpleNamespace(
        rank=1,
        lat_deg=20.1908,
        lon_deg=30.7717,
        score=0.95,
        safe_radius_m=500.0,
        zone_mean_slope_deg=1.2,
        zone_max_slope_deg=3.4,
        zone_mean_roughness_m=0.8,
        zone_hazard_pixels=0,
    )
    monkeypatch.setattr(main_mod, "load_dem_roi", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "compute_terrain_metrics", lambda *a, **k: object())
    monkeypatch.setattr(main_mod, "select_landing_sites", lambda *a, **k: SimpleNamespace(sites=[fake_site]))
    monkeypatch.setattr(main_mod, "plot_analysis_overview", lambda *a, **k: tmp_path / "overview.png")
    monkeypatch.setattr(main_mod, "plot_site_3d", lambda *a, **k: tmp_path / "site.html")

    captured: dict = {}

    def _fake_run_nac_touchdown(geotiff_path, **kwargs):
        captured["geotiff_path"] = geotiff_path
        captured.update(kwargs)
        return SimpleNamespace(
            crop=SimpleNamespace(image=np.zeros((10, 10), dtype=np.uint8), meters_per_px=0.5),
            boxes_xyxy=np.empty((0, 4), dtype=np.float32),
            hazard=SimpleNamespace(touchdown=None),
            touchdown_lat_deg=None,
            touchdown_lon_deg=None,
        )

    monkeypatch.setattr(nac_mod, "run_nac_touchdown", _fake_run_nac_touchdown)

    out_dir = tmp_path / "out"
    main_mod.main([
        "--out-dir", str(out_dir),
        "--max-slope", "7.5",
        "--nac-geotiff", str(nac_tif),
        "--nac-weights", str(weights),
        "--nac-dem", str(nac_dem),
    ])

    assert captured["geotiff_path"] == nac_tif
    assert captured["dem_path"] == nac_dem
    assert captured["max_slope_deg"] == 7.5
    assert captured["tiled"] is True
    assert captured["output_plot_path"] == out_dir / "nac_rank1_touchdown.png"
