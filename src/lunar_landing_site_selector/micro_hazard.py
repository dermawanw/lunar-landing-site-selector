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


def _tile_origins(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    origins = list(range(0, length - tile_size, stride))
    last = length - tile_size
    if not origins or origins[-1] != last:
        origins.append(last)
    return origins


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    b = np.asarray(boxes, dtype=np.float32)
    s = np.asarray(scores, dtype=np.float32)
    areas = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    order = np.argsort(s)[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(b[i, 0], b[rest, 0])
        yy1 = np.maximum(b[i, 1], b[rest, 1])
        xx2 = np.minimum(b[i, 2], b[rest, 2])
        yy2 = np.minimum(b[i, 3], b[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0.0)
        order = rest[iou <= iou_threshold]

    return b[keep], s[keep]


def _to_rgb_uint8(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        img_p = Path(image)
        if not img_p.is_file():
            raise FileNotFoundError(f"Image not found: {img_p}")
        import matplotlib.image as mpimg

        arr = mpimg.imread(str(img_p))
    else:
        arr = np.asarray(image)

    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if arr_f.size > 0 and float(np.nanmax(arr_f)) <= 1.0:
            arr_f = arr_f * 255.0
        arr = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)

    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # shortcut: passes RGB channels as-is; identical to BGR for grayscale lunar NAC
        return arr[:, :, :3]
    raise ValueError(f"Expected 2D or 3-channel image, got shape {arr.shape}")


def _predict_tiles_yolo(
    tiles: list[np.ndarray],
    weights_path: Path,
    conf: float,
    imgsz: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for tile in tiles:
        res = model.predict(source=tile, conf=conf, imgsz=imgsz, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            outputs.append((np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)))
        else:
            outputs.append(
                (
                    res.boxes.xyxy.cpu().numpy().astype(np.float32),
                    res.boxes.conf.cpu().numpy().astype(np.float32),
                )
            )
    return outputs


def detect_craters_yolo_tiled(
    image: np.ndarray | str | Path,
    weights_path: str | Path,
    conf: float = 0.25,
    tile_size: int = 640,
    overlap: int = 128,
    iou_threshold: float = 0.45,
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    """Jalankan YOLO dengan sliding-window tiles + IoU NMS untuk citra NAC beresolusi tinggi."""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < tile_size, got {overlap}")
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    rgb = _to_rgb_uint8(image)
    h, w = rgb.shape[:2]
    ys = _tile_origins(h, tile_size, overlap)
    xs = _tile_origins(w, tile_size, overlap)

    tiles: list[np.ndarray] = []
    offsets: list[tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            tiles.append(rgb[y0 : min(y0 + tile_size, h), x0 : min(x0 + tile_size, w)])
            offsets.append((x0, y0))

    raw_preds = _predict_tiles_yolo(tiles, Path(weights_path), conf=conf, imgsz=tile_size)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for (x0, y0), (boxes, scores) in zip(offsets, raw_preds):
        if len(boxes) == 0:
            continue
        shifted = boxes.astype(np.float32).copy()
        shifted[:, [0, 2]] += float(x0)
        shifted[:, [1, 3]] += float(y0)
        all_boxes.append(shifted)
        all_scores.append(scores.astype(np.float32))

    if not all_boxes:
        return (h, w), np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    merged_boxes, merged_scores = _nms_xyxy(
        np.concatenate(all_boxes, axis=0),
        np.concatenate(all_scores, axis=0),
        iou_threshold=iou_threshold,
    )
    return (h, w), merged_boxes, merged_scores


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
