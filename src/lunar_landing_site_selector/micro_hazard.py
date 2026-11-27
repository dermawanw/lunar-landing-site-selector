from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class MicroTouchdown:
    row: int
    col: int
    clearance_px: float
    clearance_m: float


@dataclass
class MicroHazardResult:
    hazard_mask: np.ndarray     # 2D bool: True = kawah/bahaya
    clearance_m: np.ndarray     # 2D float32: jarak ke bahaya terdekat (meter)
    touchdown: MicroTouchdown | None


def rasterize_crater_hazards(
    image_shape: tuple[int, int],
    boxes_xyxy: np.ndarray,
    rim_margin_factor: float = 1.1,
) -> np.ndarray:
    """Konversi bounding box [x1, y1, x2, y2] menjadi masker bahaya elips."""
    h, w = image_shape
    if h <= 0 or w <= 0:
        raise ValueError(f"image_shape must be positive, got {image_shape}")

    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if boxes.size == 0:
        return np.zeros((h, w), dtype=bool)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"boxes_xyxy must have shape (N, 4), got {boxes.shape}")

    # shortcut: rasterizes YOLO boxes as axis-aligned ellipses with uniform rim_margin_factor;
    # revisit if switching to rotated boxes or pixel-level segmentation
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        rx = max(0.5 * abs(x2 - x1), 0.5) * rim_margin_factor
        ry = max(0.5 * abs(y2 - y1), 0.5) * rim_margin_factor

        r0 = max(0, int(np.floor(cy - ry)))
        r1 = min(h, int(np.ceil(cy + ry)) + 1)
        c0 = max(0, int(np.floor(cx - rx)))
        c1 = min(w, int(np.ceil(cx + rx)) + 1)
        if r0 >= r1 or c0 >= c1:
            continue

        yy, xx = np.ogrid[r0:r1, c0:c1]
        inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        mask[r0:r1, c0:c1] |= inside

    return mask


def select_pinpoint_touchdown(
    hazard_mask: np.ndarray,
    meters_per_px: float = 1.0,
) -> MicroHazardResult:
    """Cari titik touchdown paling jauh dari kawah mikro dan tepi frame."""
    if meters_per_px <= 0:
        raise ValueError(f"meters_per_px must be > 0, got {meters_per_px}")

    # shortcut: assumes square pixels (uniform meters_per_px) on an orthographic crop;
    # revisit when integrating georeferenced GeoTIFF LROC NAC tiles
    hazard = np.asarray(hazard_mask, dtype=bool)
    bounded_safe = ~hazard.copy()
    bounded_safe[0, :] = False
    bounded_safe[-1, :] = False
    bounded_safe[:, 0] = False
    bounded_safe[:, -1] = False

    clearance_px = distance_transform_edt(bounded_safe).astype(np.float32)
    clearance_m = (clearance_px * meters_per_px).astype(np.float32)

    if float(clearance_px.max()) <= 0.0:
        return MicroHazardResult(hazard_mask=hazard, clearance_m=clearance_m, touchdown=None)

    best_r, best_c = np.unravel_index(int(np.argmax(clearance_px)), clearance_px.shape)
    touchdown = MicroTouchdown(
        row=int(best_r),
        col=int(best_c),
        clearance_px=float(clearance_px[best_r, best_c]),
        clearance_m=float(clearance_m[best_r, best_c]),
    )
    return MicroHazardResult(hazard_mask=hazard, clearance_m=clearance_m, touchdown=touchdown)


def detect_craters_yolo(
    image_path: str | Path,
    weights_path: str | Path,
    conf: float = 0.25,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Jalankan inferensi YOLO; ultralytics di-import lazy agar CI tanpa GPU tetap jalan."""
    img_p = Path(image_path)
    w_p = Path(weights_path)
    if not img_p.is_file():
        raise FileNotFoundError(f"Image not found: {img_p}")
    if not w_p.is_file():
        raise FileNotFoundError(f"Weights not found: {w_p}")

    from ultralytics import YOLO

    model = YOLO(str(w_p))
    result = model.predict(source=str(img_p), conf=conf, verbose=False)[0]
    h, w = result.orig_shape
    if result.boxes is None or len(result.boxes) == 0:
        return (h, w), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(np.float32)
    return (h, w), boxes, scores


def save_micro_hazard_plot(
    image_path: str | Path,
    boxes_xyxy: np.ndarray,
    result: MicroHazardResult,
    output_path: str | Path,
    conf: float = 0.20,
    meters_per_px: float | None = None,
) -> Path:
    """Simpan visualisasi 3-panel: deteksi YOLO, hazard mask, dan pinpoint touchdown."""
    import matplotlib.image as mpimg
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Rectangle

    img = mpimg.imread(str(image_path))
    cmap_img = "gray" if img.ndim == 2 else None
    unit = "m" if meters_per_px is not None else "px"
    clearance_grid = result.clearance_m if meters_per_px is not None else result.clearance_m

    fig = Figure(figsize=(15, 5), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # Panel 1: Deteksi kawah YOLO
    axes[0].imshow(img, cmap=cmap_img)
    for x1, y1, x2, y2 in boxes_xyxy:
        axes[0].add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.2, edgecolor="#ffd166", facecolor="none")
        )
    axes[0].set_title(f"1. Crater Detections ({len(boxes_xyxy)}, conf>={conf:.2f})")

    # Panel 2: Rasterized micro-hazard mask
    axes[1].imshow(img, cmap=cmap_img)
    overlay = np.zeros((*result.hazard_mask.shape, 4), dtype=np.float32)
    overlay[result.hazard_mask] = [1.0, 0.2, 0.2, 0.45]
    axes[1].imshow(overlay)
    hazard_pct = float(result.hazard_mask.mean() * 100.0)
    axes[1].set_title(f"2. Micro-Hazard Mask ({hazard_pct:.1f}% blocked)")

    # Panel 3: EDT clearance map & pinpoint touchdown
    im = axes[2].imshow(clearance_grid, cmap="viridis")
    if result.touchdown is not None:
        td = result.touchdown
        val = td.clearance_m if meters_per_px is not None else td.clearance_px
        axes[2].add_patch(
            Circle(
                (td.col, td.row),
                radius=td.clearance_px,
                edgecolor="#00f5d4",
                facecolor="none",
                linewidth=1.8,
                linestyle="--",
            )
        )
        axes[2].scatter([td.col], [td.row], marker="*", s=220, c="#00f5d4", edgecolors="black", linewidths=0.8, zorder=5)
        axes[2].set_title(f"3. Pinpoint Touchdown (r={val:.1f} {unit})")
    else:
        axes[2].set_title("3. No Safe Touchdown Found")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label=f"Clearance ({unit})")

    for ax in axes:
        ax.set_axis_off()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2: YOLO crater detection & pinpoint touchdown selector.")
    parser.add_argument("image", type=Path, help="Path to top-down lunar optical image.")
    parser.add_argument("--weights", type=Path, default=Path("runs/detect/crater_v1/weights/best.pt"))
    parser.add_argument("--conf", type=float, default=0.20, help="Recall-biased threshold for landing safety.")
    parser.add_argument("--rim-margin", type=float, default=1.1, help="Safety expansion factor around crater rims.")
    parser.add_argument("--meters-per-px", type=float, default=None, help="Optional ground sampling distance (m/px).")
    parser.add_argument("--output", type=Path, default=Path("outputs/micro_hazard_analysis.png"))
    args = parser.parse_args()

    shape, boxes, _ = detect_craters_yolo(args.image, args.weights, conf=args.conf)
    mask = rasterize_crater_hazards(shape, boxes, rim_margin_factor=args.rim_margin)
    # shortcut: uses 1.0 when image GSD is unknown so EDT values stay in pixel units
    scale = args.meters_per_px if args.meters_per_px is not None else 1.0
    result = select_pinpoint_touchdown(mask, meters_per_px=scale)
    saved = save_micro_hazard_plot(
        args.image, boxes, result, args.output, conf=args.conf, meters_per_px=args.meters_per_px
    )

    print(f"Detected craters : {len(boxes)} (conf >= {args.conf})")
    print(f"Hazard coverage  : {mask.mean() * 100:.1f}%")
    if result.touchdown is None:
        print("Touchdown        : None (frame fully blocked)")
    else:
        td = result.touchdown
        clearance_str = f"{td.clearance_m:.1f} m" if args.meters_per_px else f"{td.clearance_px:.1f} px"
        print(f"Touchdown (r, c) : ({td.row}, {td.col}) | Clearance: {clearance_str}")
    print(f"Saved plot       -> {saved}")
