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
