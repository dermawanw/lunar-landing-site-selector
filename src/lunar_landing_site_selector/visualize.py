from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import plotly.graph_objects as go

from lunar_landing_site_selector.ingest import LunarDEM
from lunar_landing_site_selector.site_selector import LandingSite, SelectionResult
from lunar_landing_site_selector.terrain_metrics import TerrainMetrics


def plot_analysis_overview(
    dem: LunarDEM,
    metrics: TerrainMetrics,
    result: SelectionResult,
    output_path: str | Path,
    min_safe_radius_m: float = 500.0,
    zoom_half_px: int = 60,
) -> Path:
    """Buat peta multi-panel 2x2 (Elevasi, Slope, Suitability, Zoom Rank #1)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lon_min, lat_min, lon_max, lat_max = dem.bounds
    extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=180)
    fig.suptitle("Lunar Landing Site Analysis — SLDEM2015 (512 ppd)", fontsize=15, fontweight="bold")

    # 1. Elevasi (DEM)
    im0 = axes[0, 0].imshow(dem.elevation_m, cmap="terrain", extent=extent, origin="upper")
    axes[0, 0].set_title("1. Elevation (DEM)")
    axes[0, 0].set_ylabel("Latitude (°N)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="Meters")

    # 2. Slope (dibatasi 0-25 deg agar kontras kawah & dataran jelas)
    im1 = axes[0, 1].imshow(metrics.slope_deg, cmap="inferno", vmin=0, vmax=25, extent=extent, origin="upper")
    axes[0, 1].set_title("2. Terrain Slope")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="Degrees (°)")

    # 3. Peta Kelayakan + Pin Top-N
    axes[1, 0].imshow(np.where(result.safe_mask, 1.0, 0.0), cmap="gray", vmin=0, vmax=1.6, extent=extent, origin="upper")
    im2 = axes[1, 0].imshow(result.score_map, cmap="turbo", vmin=0, vmax=1, extent=extent, origin="upper")
    for s in result.sites:
        axes[1, 0].plot(s.lon_deg, s.lat_deg, marker="*", markersize=12, color="white", markeredgecolor="black")
        axes[1, 0].text(
            s.lon_deg + 0.06, s.lat_deg + 0.06, f"#{s.rank}",
            color="white", fontsize=10, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.65, pad=1.5, edgecolor="none"),
        )
    axes[1, 0].set_title("3. Suitability Score & Top Candidates")
    axes[1, 0].set_xlabel("Longitude (°E)")
    axes[1, 0].set_ylabel("Latitude (°N)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Composite Score")

    # 4. Zoom pada Rank #1 (koordinat meter lokal -> lingkaran bebas distorsi)
    ax_z = axes[1, 1]
    if result.sites:
        top = result.sites[0]
        h, w = dem.elevation_m.shape
        r0, r1 = max(0, top.row - zoom_half_px), min(h, top.row + zoom_half_px + 1)
        c0, c1 = max(0, top.col - zoom_half_px), min(w, top.col + zoom_half_px + 1)

        slope_patch = metrics.slope_deg[r0:r1, c0:c1]
        hazard_patch = ~result.safe_mask[r0:r1, c0:c1]

        x_min = (c0 - top.col - 0.5) * dem.dx_m
        x_max = (c1 - top.col - 0.5) * dem.dx_m
        y_max = (top.row - r0 + 0.5) * dem.dy_m
        y_min = (top.row - r1 + 0.5) * dem.dy_m

        im3 = ax_z.imshow(slope_patch, cmap="viridis", vmin=0, vmax=15, extent=[x_min, x_max, y_min, y_max], origin="upper")
        # Overlay merah transparan untuk piksel bahaya
        hazard_rgba = np.zeros((*hazard_patch.shape, 4), dtype=np.float32)
        hazard_rgba[hazard_patch] = [1.0, 0.15, 0.15, 0.45]
        ax_z.imshow(hazard_rgba, extent=[x_min, x_max, y_min, y_max], origin="upper")

        ax_z.add_patch(Circle((0, 0), min_safe_radius_m, fill=False, edgecolor="cyan", linewidth=2.0, label=f"Target zone ({min_safe_radius_m:.0f} m)"))
        ax_z.add_patch(Circle((0, 0), top.safe_radius_m, fill=False, edgecolor="lime", linestyle="--", linewidth=1.8, label=f"Clearance ({top.safe_radius_m:.0f} m)"))
        ax_z.plot(0, 0, marker="+", markersize=10, color="white", markeredgewidth=2)
        ax_z.set_aspect("equal")
        ax_z.set_title(f"4. Zoom Rank #1 ({top.lat_deg:.4f}°N, {top.lon_deg:.4f}°E)")
        ax_z.set_xlabel("East–West Offset (m)")
        ax_z.set_ylabel("North–South Offset (m)")
        ax_z.legend(loc="upper right", fontsize=8, framealpha=0.85)
        fig.colorbar(im3, ax=ax_z, fraction=0.046, pad=0.04, label="Slope (°) | Red = Hazard")
    else:
        ax_z.text(0.5, 0.5, "No valid site found", ha="center", va="center")
        ax_z.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_site_3d(
    dem: LunarDEM,
    site: LandingSite,
    output_path: str | Path,
    min_safe_radius_m: float = 500.0,
    half_px: int = 60,
) -> Path:
    """Ekspor permukaan 3D interaktif (HTML) di sekitar kandidat terpilih."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = dem.elevation_m.shape
    r0, r1 = max(0, site.row - half_px), min(h, site.row + half_px + 1)
    c0, c1 = max(0, site.col - half_px), min(w, site.col + half_px + 1)

    z_patch = dem.elevation_m[r0:r1, c0:c1]
    xs = (np.arange(c0, c1) - site.col) * dem.dx_m
    ys = (site.row - np.arange(r0, r1)) * dem.dy_m

    # Lingkaran zona pendaratan 500 m yang menempel di permukaan
    theta = np.linspace(0, 2 * np.pi, 120)
    rx = min_safe_radius_m * np.cos(theta)
    ry = min_safe_radius_m * np.sin(theta)
    ring_cols = np.clip(np.round(rx / dem.dx_m).astype(int) + (site.col - c0), 0, z_patch.shape[1] - 1)
    ring_rows = np.clip(np.round(-ry / dem.dy_m).astype(int) + (site.row - r0), 0, z_patch.shape[0] - 1)
    rz = z_patch[ring_rows, ring_cols] + 8.0

    z_center = float(dem.elevation_m[site.row, site.col])

    fig = go.Figure(
        data=[
            go.Surface(x=xs, y=ys, z=z_patch, colorscale="Earth", colorbar=dict(title="Elevation (m)")),
            go.Scatter3d(
                x=rx, y=ry, z=rz, mode="lines",
                line=dict(color="cyan", width=5),
                name=f"Landing Zone ({min_safe_radius_m:.0f} m radius)",
            ),
            go.Scatter3d(
                x=[0.0], y=[0.0], z=[z_center + 20.0], mode="markers",
                marker=dict(size=6, color="red"),
                name=f"Rank #{site.rank} Center",
            ),
        ]
    )

    fig.update_layout(
        title=f"3D Terrain Inspection — Rank #{site.rank} ({site.lat_deg:.4f}°N, {site.lon_deg:.4f}°E) [Vertical Exaggeration]",
        legend=dict(x=0.01, y=0.98, bgcolor="rgba(255, 255, 255, 0.85)"),
        scene=dict(
            xaxis_title="East–West (m)",
            yaxis_title="North–South (m)",
            zaxis=dict(title="Elevation (m)", nticks=4),
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.15),
        ),
        margin=dict(l=0, r=0, b=0, t=45),
    )

    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path
