import argparse
from pathlib import Path

from lunar_landing_site_selector.ingest import load_dem_roi
from lunar_landing_site_selector.site_selector import select_landing_sites
from lunar_landing_site_selector.terrain_metrics import compute_terrain_metrics
from lunar_landing_site_selector.visualize import plot_analysis_overview, plot_site_3d


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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


if __name__ == "__main__":
    main()
