"""The thermal calibration must recover arbitrary mountings (no built-in assumptions about the pose)."""

import numpy as np
import pytest

from segkit import thermal_calib as tc


def synthetic(p_true: np.ndarray, mirror: bool, n: int = 90, seed: int = 0):
    """Ellipse 'hands' at random places/depths in the camera crop, seen by a thermal camera at p_true."""
    rng = np.random.default_rng(seed)
    f, cx, cy, box = 460.0, 240.0, 320.0, (24, 104, 432)
    s = tc.Session(f, cx, cy, np.zeros(0), np.zeros((0, 24, 32)), np.zeros(0), [], None)
    yy, xx = np.mgrid[0:tc.MASK_GRID, 0:tc.MASK_GRID]
    masks, depth, cam_pt = [], [], []
    for _ in range(n):
        z = rng.uniform(18, 45)
        r_px = f * 5.5 / z / box[2] * tc.MASK_GRID            # ~11 cm wide hand, in mask cells
        gx, gy = rng.uniform(r_px, tc.MASK_GRID - r_px, 2)
        m = (((xx - gx) / r_px) ** 2 + ((yy - gy) / (1.6 * r_px)) ** 2 <= 1).astype(np.float32)
        masks.append(m)
        depth.append(z)
        px, py = box[0] + (gx + 0.5) / tc.MASK_GRID * box[2], box[1] + (gy + 0.5) / tc.MASK_GRID * box[2]
        cam_pt.append(((px - cx) / f * z, (py - cy) / f * z, z))
    pairs = tc.Pairs(np.array(masks), np.tile(box, (n, 1)), np.array(depth), np.zeros((n, 768)),
                     np.array(cam_pt), np.zeros((n, 2)), np.arange(n), np.arange(n))
    frac, valid = tc.predict(p_true, mirror, s, pairs)
    warm = np.where(valid, frac, 0) + rng.normal(0, 0.03, frac.shape)
    keep, th_pt = [], []
    for i, w in enumerate(warm):
        c = tc.blob_centroid(w.reshape(24, 32), 0.5)
        if c is not None:
            keep.append(i)
            th_pt.append(c)
    pairs.warm = warm
    pairs.th_pt = np.zeros((n, 2))
    pairs.th_pt[keep] = th_pt
    return s, pairs.every(1) if not keep else tc.Pairs(*(a[keep] for a in vars(pairs).values()))


@pytest.mark.parametrize("deg, cm, mirror", [
    ((-30, 5, 45), (20, -3, 1), True),     # 30 deg yaw, 45 deg roll, 20 cm to the side
    ((2, -3, 92), (0, 3.5, -1), True),     # close to the real phone mounting
    ((10, 20, -60), (-8, 10, 0), False),
])
def test_recovers_arbitrary_mounting(deg, cm, mirror):
    p_true = np.array([*np.radians(deg), *cm, np.log(1.05)])
    s, pairs = synthetic(p_true, mirror)
    assert len(pairs.depth) > 30, "the synthetic thermal camera must see the hand in most frames"
    p1, m1, err = tc.pose_from_points(pairs)
    assert m1 == mirror
    p = tc.refine(p1, m1, s, pairs)
    ang_err = np.degrees(np.abs((p[:3] - p_true[:3] + np.pi) % (2 * np.pi) - np.pi))
    assert ang_err.max() < 1.5, ang_err
    assert np.abs(p[3:6] - p_true[3:6]).max() < 1.5, p[3:6]
    assert abs(np.exp(p[6]) - 1.05) < 0.06   # weakly observable, datasheet prior
