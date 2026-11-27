import numpy as np
import pytest

from lunar_landing_site_selector.micro_hazard import (
    rasterize_crater_hazards,
    select_pinpoint_touchdown,
)


def test_empty_detections_picks_frame_center():
    mask = rasterize_crater_hazards((101, 101), np.empty((0, 4)))
    result = select_pinpoint_touchdown(mask, meters_per_px=0.5)

    assert result.touchdown is not None
    assert (result.touchdown.row, result.touchdown.col) == (50, 50)
    assert result.touchdown.clearance_m == pytest.approx(25.0)


def test_crater_mask_avoids_hazard_and_scales_clearance():
    boxes = np.array([[30.0, 30.0, 70.0, 70.0]])
    mask = rasterize_crater_hazards((101, 101), boxes, rim_margin_factor=1.1)
    result = select_pinpoint_touchdown(mask, meters_per_px=0.5)

    assert mask[50, 50]
    assert result.touchdown is not None
    assert not mask[result.touchdown.row, result.touchdown.col]
    assert result.touchdown.clearance_m == pytest.approx(result.touchdown.clearance_px * 0.5)


def test_fully_blocked_frame_returns_none_touchdown():
    boxes = np.array([[0.0, 0.0, 50.0, 50.0]])
    mask = rasterize_crater_hazards((50, 50), boxes, rim_margin_factor=1.5)
    result = select_pinpoint_touchdown(mask, meters_per_px=1.0)

    assert result.touchdown is None


def test_save_micro_hazard_plot_writes_png(tmp_path):
    import matplotlib.image as mpimg
    from lunar_landing_site_selector.micro_hazard import save_micro_hazard_plot

    img_path = tmp_path / "synthetic_lunar.png"
    mpimg.imsave(img_path, np.zeros((64, 64), dtype=np.uint8), cmap="gray")

    boxes = np.array([[10.0, 10.0, 26.0, 26.0]])
    mask = rasterize_crater_hazards((64, 64), boxes)
    result = select_pinpoint_touchdown(mask, meters_per_px=1.0)

    out_path = save_micro_hazard_plot(img_path, boxes, result, tmp_path / "out.png")
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


def test_nms_merges_overlapping_boxes_and_keeps_distinct():
    from lunar_landing_site_selector.micro_hazard import _nms_xyxy

    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],   # high score
            [12.0, 12.0, 52.0, 52.0],   # heavy overlap -> should drop
            [100.0, 100.0, 140.0, 140.0],  # distinct -> should keep
        ],
        dtype=np.float32,
    )
    scores = np.array([0.90, 0.70, 0.80], dtype=np.float32)

    kept_boxes, kept_scores = _nms_xyxy(boxes, scores, iou_threshold=0.45)
    assert len(kept_boxes) == 2
    assert kept_scores.tolist() == pytest.approx([0.90, 0.80])


def test_tiled_inference_offsets_and_deduplicates_overlap(tmp_path, monkeypatch):
    import lunar_landing_site_selector.micro_hazard as mh

    # 800x800 image with tile_size=640, overlap=128 -> 2x2 tiles at origins [0, 160]
    img = np.zeros((800, 800), dtype=np.uint8)

    def _fake_predict(tiles, weights_path, conf, imgsz):
        assert len(tiles) == 4
        # Tile (0,0) and Tile (160,0) both see the same crater at global x=[200..260], y=[100..160]
        return [
            (np.array([[200.0, 100.0, 260.0, 160.0]], dtype=np.float32), np.array([0.92], dtype=np.float32)),
            (np.array([[42.0, 100.0, 102.0, 160.0]], dtype=np.float32), np.array([0.81], dtype=np.float32)),
            (np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)),
            (np.array([[400.0, 400.0, 460.0, 460.0]], dtype=np.float32), np.array([0.85], dtype=np.float32)),
        ]

    monkeypatch.setattr(mh, "_predict_tiles_yolo", _fake_predict)
    shape, boxes, scores = mh.detect_craters_yolo_tiled(
        img,
        tmp_path / "dummy.pt",
        tile_size=640,
        overlap=128,
        iou_threshold=0.45,
    )

    assert shape == (800, 800)
    assert len(boxes) == 2  # overlap duplicate removed, bottom-right tile shifted by (+160, +160)
    assert scores.tolist() == pytest.approx([0.92, 0.85])
    assert boxes[1].tolist() == pytest.approx([560.0, 560.0, 620.0, 620.0])
