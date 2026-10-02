"""Boundary accuracy between a predicted and a ground-truth hand mask, in full-res pixels."""

from dataclasses import dataclass

import cv2
import numpy as np

F_THRESHOLDS_PX = (1, 2, 4)


@dataclass
class Scores:
    iou: float
    p50_px: float              # median distance, both directions pooled
    p95_px: float
    max_px: float
    f_at: dict[int, float]     # boundary F-score at each tolerance in px


def boundary(mask: np.ndarray) -> np.ndarray:
    """Inner boundary: mask pixels with at least one 4-neighbour outside the mask."""
    m = mask.astype(np.uint8)
    return mask & ~cv2.erode(m, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def distance_to(target: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance to the nearest True pixel of `target`."""
    if not target.any():
        return np.full(target.shape, np.inf, np.float32)
    return cv2.distanceTransform((~target).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def score(pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray | None = None) -> Scores:
    pred, gt = pred.astype(bool), gt.astype(bool)
    keep = np.ones_like(gt) if ignore is None else ~ignore.astype(bool)
    union = (pred | gt) & keep
    iou = float(((pred & gt) & keep).sum() / union.sum()) if union.any() else 1.0

    bp, bg = boundary(pred), boundary(gt)
    d_pred = distance_to(bg)[bp & keep]  # each predicted edge pixel -> nearest true edge
    d_gt = distance_to(bp)[bg & keep]    # each true edge pixel -> nearest predicted edge
    # An empty prediction (or label) has no edge to measure against: count it as the worst possible error.
    diagonal = float(np.hypot(*gt.shape))
    d_pred, d_gt = np.minimum(d_pred, diagonal), np.minimum(d_gt, diagonal)
    pooled = np.concatenate([d_pred, d_gt])
    if pooled.size == 0:
        return Scores(iou, 0.0, 0.0, 0.0, {t: 1.0 for t in F_THRESHOLDS_PX})

    f_at = {}
    for t in F_THRESHOLDS_PX:
        precision = float((d_pred <= t).mean()) if d_pred.size else 0.0
        recall = float((d_gt <= t).mean()) if d_gt.size else 0.0
        f_at[t] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return Scores(iou, float(np.percentile(pooled, 50)), float(np.percentile(pooled, 95)),
                  float(pooled.max()), f_at)
