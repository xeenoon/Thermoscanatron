import json

import cv2
import numpy as np
import pytest

from segkit.eval.labels import load_label, rasterize
from segkit.eval.metrics import score


def disk(shape, center, r):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, center, r, 1, -1)
    return m.astype(bool)


def test_perfect_prediction():
    gt = disk((200, 200), (100, 100), 50)
    s = score(gt, gt)
    assert s.iou == 1.0 and s.p95_px == 0.0 and s.f_at[1] == 1.0


def test_shift_measured_in_pixels():
    gt = disk((200, 200), (100, 100), 50)
    pred = disk((200, 200), (103, 100), 50)  # 3px shift: max error 3px at left/right, 0 at top/bottom
    s = score(pred, gt)
    assert 2.0 <= s.p95_px <= 3.2
    assert s.max_px == pytest.approx(3.0, abs=0.1)
    assert s.f_at[4] == 1.0 and s.f_at[1] < 1.0


def test_dilation_by_k_pixels():
    gt = disk((300, 300), (150, 150), 80)
    pred = disk((300, 300), (150, 150), 82)
    s = score(pred, gt)
    assert s.p50_px == pytest.approx(2.0, abs=0.5)
    assert s.f_at[1] < s.f_at[2] <= s.f_at[4] == 1.0


def test_missed_finger_dominates_p95_not_iou():
    gt = disk((300, 300), (150, 150), 60)
    cv2.rectangle(gt_u8 := gt.astype(np.uint8), (145, 20), (155, 150), 1, -1)  # thin "finger"
    gt = gt_u8.astype(bool)
    pred = disk((300, 300), (150, 150), 60)
    s = score(pred, gt)
    assert s.iou > 0.9           # a finger is a small fraction of area...
    assert s.p95_px > 20         # ...but a huge boundary error
    assert s.f_at[2] < 0.95


def test_ignore_region_excluded():
    gt = disk((200, 200), (100, 100), 50)
    pred = gt.copy()
    pred[140:, :] = False         # chop the bottom off, like a different wrist cut
    ignore = np.zeros_like(gt)
    ignore[130:, :] = True
    assert score(pred, gt).p95_px > 5
    assert score(pred, gt, ignore).p95_px == 0.0


def test_labelme_pixel_corner_convention():
    # LabelMe square from corner (10,10) to (20,20) covers pixels 10..19 exactly.
    sq = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], float)
    m = rasterize([sq], (32, 32))
    ys, xs = np.nonzero(m)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (10, 19, 10, 19)
    assert m.sum() == 100


def test_load_label_with_hole_and_ignore(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), np.zeros((100, 120, 3), np.uint8))
    shapes = [
        {"label": "hand", "points": [[10, 10], [90, 10], [90, 90], [10, 90]], "shape_type": "polygon"},
        {"label": "hole", "points": [[40, 40], [60, 40], [60, 60], [40, 60]], "shape_type": "polygon"},
        {"label": "ignore", "points": [[0, 80], [120, 80], [120, 100], [0, 100]], "shape_type": "polygon"},
    ]
    (tmp_path / "a.json").write_text(json.dumps(
        {"shapes": shapes, "imagePath": "a.jpg", "imageHeight": 100, "imageWidth": 120}))
    lab = load_label(tmp_path / "a.json")
    assert lab.hand.sum() == 80 * 80 - 20 * 20
    assert not lab.hand[50, 50] and lab.hand[20, 20]
    assert lab.ignore.sum() == 20 * 120
    assert lab.image_path.exists()


def test_empty_prediction_is_worst_case_not_nan():
    gt = disk((100, 100), (50, 50), 20)
    s = score(np.zeros_like(gt), gt)
    assert s.iou == 0.0 and s.f_at[2] == 0.0
    assert s.p95_px == pytest.approx(np.hypot(100, 100))
