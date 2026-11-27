import argparse
from pathlib import Path

from lunar_landing_site_selector.ingest import load_dem_roi
from lunar_landing_site_selector.site_selector import select_landing_sites
from lunar_landing_site_selector.terrain_metrics import compute_terrain_metrics
from lunar_landing_site_selector.visualize import plot_analysis_overview, plot_site_3d


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lunar Landing Site Selector Pipeline")
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path("data/raw/SLDEM2015_512_00N_30N_000_045.JP2"),
        help="Path ke file DEM SLDEM2015",
    )
    parser.add_argument("--scale", type=float, default=0.5, help="Faktor skala elevasi ke meter")
    parser.add_argument("--max-slope", type=float, default=5.0, help="Batas maksimum slope (derajat)")
    parser.add_argument("--max-roughness", type=float, default=5.0, help="Batas maksimum roughness (meter)")
    parser.add_argument("--min-radius", type=float, default=500.0, help="Radius zona aman minimum (meter)")
    parser.add_argument("--top-n", type=int, default=3, help="Jumlah kandidat pendaratan")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Folder output visualisasi")
    parser.add_argument("--nac-geotiff", type=Path, default=None, help="Opsional: path GeoTIFF LROC NAC untuk refinement Rank #1")
    parser.add_argument("--nac-weights", type=Path, default=None, help="Opsional: path bobot YOLO (best.pt) untuk deteksi kawah pada crop NAC")
    parser.add_argument("--nac-dem", type=Path, default=None, help="Opsional: path DEM resolusi tinggi untuk fusi slope pada crop NAC")
    args = parser.parse_args(argv)
    if args.nac_dem is not None and (args.nac_geotiff is None or args.nac_weights is None):
        parser.error("--nac-dem membutuhkan --nac-geotiff dan --nac-weights")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    print(f"[1/4] Loading DEM ROI: {args.dem}")
    dem = load_dem_roi(args.dem, window=(0, 0, 2048, 2048), scale_to_meters=args.scale)

    print("[2/4] Computing slope & roughness...")
    metrics = compute_terrain_metrics(dem, window_px=5)

    print("[3/4] Selecting optimal landing sites...")
    result = select_landing_sites(
        dem,
        metrics,
        max_slope_deg=args.max_slope,
        max_roughness_m=args.max_roughness,
        min_safe_radius_m=args.min_radius,
        top_n=args.top_n,
    )

    print("[4/4] Generating 2D & 3D visualizations...")
    overview_path = plot_analysis_overview(
        dem, metrics, result, args.out_dir / "landing_site_analysis.png", min_safe_radius_m=args.min_radius
    )
    print(f"Saved 2D overview -> {overview_path}")

    if result.sites:
        html_path = plot_site_3d(
            dem, result.sites[0], args.out_dir / "site_rank1_3d.html", min_safe_radius_m=args.min_radius
        )
        print(f"Saved 3D terrain  -> {html_path}")

    print("\n| Rank | Latitude (°N) | Longitude (°E) | Score | Clearance (m) | Mean / Max Slope (°) | Mean Roughness (m) | Hazards |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in result.sites:
        print(
            f"| #{s.rank} | {s.lat_deg:.4f} | {s.lon_deg:.4f} | {s.score:.3f} | "
            f"{s.safe_radius_m:.0f} | {s.zone_mean_slope_deg:.2f} / {s.zone_max_slope_deg:.2f} | "
            f"{s.zone_mean_roughness_m:.2f} | {s.zone_hazard_pixels} |"
        )

    if args.nac_geotiff is not None and result.sites:
        if not args.nac_geotiff.exists():
            raise FileNotFoundError(f"NAC GeoTIFF tidak ditemukan: {args.nac_geotiff}")
        from lunar_landing_site_selector.nac_georegister import extract_nac_window, run_nac_touchdown

        top_site = result.sites[0]
        print(f"\n[Phase 3] Georeferencing NAC around Rank #1 ({top_site.lat_deg:.4f}°N, {top_site.lon_deg:.4f}°E)...")
        if args.nac_weights is None:
            crop = extract_nac_window(
                args.nac_geotiff,
                lat_deg=top_site.lat_deg,
                lon_deg=top_site.lon_deg,
                half_span_m=args.min_radius,
            )
            c_lat, c_lon = crop.pixel_to_latlon(crop.image.shape[0] / 2.0 - 0.5, crop.image.shape[1] / 2.0 - 0.5)
            print(
                f"NAC crop: {crop.image.shape[1]}x{crop.image.shape[0]} px "
                f"({crop.meters_per_px:.2f} m/px) | center: {c_lat:.6f}°N, {c_lon:.6f}°E"
            )
        else:
            if not args.nac_weights.exists():
                raise FileNotFoundError(f"YOLO weights tidak ditemukan: {args.nac_weights}")
            if args.nac_dem is not None and not args.nac_dem.exists():
                raise FileNotFoundError(f"NAC DEM tidak ditemukan: {args.nac_dem}")
            nac_plot_path = args.out_dir / "nac_rank1_touchdown.png"
            # shortcut: assumes --nac-dem is already in meters (dem_scale=1.0);
            # add --nac-dem-scale if fusing non-meter high-res DEMs
            report = run_nac_touchdown(
                args.nac_geotiff,
                lat_deg=top_site.lat_deg,
                lon_deg=top_site.lon_deg,
                weights_path=args.nac_weights,
                half_span_m=args.min_radius,
                tmp_image_path=args.out_dir / "_nac_crop_tmp.png",
                output_plot_path=nac_plot_path,
                tiled=True,
                dem_path=args.nac_dem,
                max_slope_deg=args.max_slope,
            )
            print(f"Saved NAC plot    -> {nac_plot_path}")
            td = report.hazard.touchdown
            print(
                f"NAC crop: {report.crop.image.shape[1]}x{report.crop.image.shape[0]} px "
                f"({report.crop.meters_per_px:.2f} m/px) | craters: {len(report.boxes_xyxy)}"
            )
            if td is None:
                print("Touchdown: None (seluruh crop tertutup hazard)")
            else:
                print(
                    f"Touchdown: ({report.touchdown_lat_deg:.6f}°N, {report.touchdown_lon_deg:.6f}°E) "
                    f"| clearance: {td.clearance_m:.1f} m"
                )


if __name__ == "__main__":
    main()
