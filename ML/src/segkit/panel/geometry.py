"""Lattice geometry: projecting the panel grid through a homography, gridline evidence maps, and fitting
a homography to that evidence (used both for anchoring and for drift correction while tracking)."""

import cv2
import numpy as np
from scipy.optimize import least_squares

from segkit.panel.spec import PanelSpec

LINE_SAMPLES_PER_CELL = 12
DIAMOND_WEIGHT = 4.0


def project(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """[N, 2] panel coordinates -> [N, 2] image pixels."""
    p = np.c_[uv, np.ones(len(uv))] @ H.T
    return p[:, :2] / p[:, 2:3]


def unproject(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return project(np.linalg.inv(H), xy)


def corners(spec: PanelSpec) -> np.ndarray:
    """Cell-area corners in panel coordinates: TL, TR, BR, BL."""
    return np.array([[0, 0], [spec.cols, 0], [spec.cols, spec.rows], [0, spec.rows]], np.float64)


def lattice_points(spec: PanelSpec, per_cell: int = LINE_SAMPLES_PER_CELL, inner_cols: bool = True) -> np.ndarray:
    """Points along every gridline (including the outer boundary), in panel coordinates.

    inner_cols=False leaves out the inner column lines: the cells' thin vertical busbars look just like them,
    so a fit uses the horizontal lines, the outer edges and the diamonds instead."""
    pts = []
    for u in (range(spec.cols + 1) if inner_cols else (0, spec.cols)):
        v = np.linspace(0, spec.rows, spec.rows * per_cell + 1)
        pts.append(np.c_[np.full_like(v, u), v])
    for v in range(spec.rows + 1):
        u = np.linspace(0, spec.cols, spec.cols * per_cell + 1)
        pts.append(np.c_[u, np.full_like(u, v)])
    return np.vstack(pts)


def cell_px(H: np.ndarray, uv: tuple[float, float]) -> float:
    """Approximate size of one cell in pixels around panel point uv (geometric mean of the two sides)."""
    u, v = uv
    p = project(H, np.array([[u, v], [u + 1, v], [u, v + 1]]))
    return float(np.sqrt(np.linalg.norm(p[1] - p[0]) * np.linalg.norm(p[2] - p[0])))


def line_evidence(gray: np.ndarray, kernel: int = 15, thresh: int = 30) -> np.ndarray:
    """Bright thin lines on a dark surround (the white gaps between dark cells), as a uint8 0/255 mask."""
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel)))
    dark = cv2.blur(gray, (31, 31)) < 120
    return ((th > thresh) & dark).astype(np.uint8) * 255


def diamond_evidence(lines: np.ndarray, size: int = 7) -> np.ndarray:
    """Blobs clearly wider than a gridline: the diamonds (and some glare), from a line_evidence mask."""
    return cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))


def distance_map(lines: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(255 - lines, cv2.DIST_L2, 3)


def _sample(dt: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return cv2.remap(dt, xy[:, 0:1].astype(np.float32), xy[:, 1:2].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)[:, 0]


def fit_to_lines(H0: np.ndarray, dt: np.ndarray, spec: PanelSpec, dt_diamond: np.ndarray | None = None,
                 trunc: float = 8.0, max_iter: int = 30) -> tuple[np.ndarray, float, int]:
    """Refine H so the projected lattice sits on the evidence. Returns (H, mean residual px, points used).

    Residuals: distance-transform value at each projected gridline point (horizontal lines and outer edges),
    plus, if dt_diamond is given, at each diamond (weighted like a few line points), all truncated at
    `trunc` so features the evidence missed (glare, occlusion) don't drag the fit.
    """
    h, w = dt.shape
    line_uv = lattice_points(spec, inner_cols=dt_diamond is None)
    dia_uv = np.array(spec.diamond_points) if dt_diamond is not None else np.zeros((0, 2))

    def inside(xy):
        return (xy[:, 0] >= 1) & (xy[:, 0] < w - 2) & (xy[:, 1] >= 1) & (xy[:, 1] < h - 2)

    line_uv = line_uv[inside(project(H0, line_uv))]
    dia_uv = dia_uv[inside(project(H0, dia_uv))] if len(dia_uv) else dia_uv
    if len(line_uv) < 20:
        return H0, float("inf"), len(line_uv)
    maps = [(np.minimum(dt, trunc), line_uv, 1.0)]
    if len(dia_uv):
        maps.append((np.minimum(dt_diamond, trunc), dia_uv, DIAMOND_WEIGHT))
    p0 = (H0 / H0[2, 2]).ravel()[:8]

    def resid(p):
        H = np.append(p, 1.0).reshape(3, 3)
        out = []
        for m, uv, wgt in maps:
            xy = project(H, uv)
            r = _sample(m, np.clip(xy, [0, 0], [w - 1, h - 1]))
            r[~inside(xy)] = trunc
            out.append(r * wgt)
        return np.concatenate(out)

    sol = least_squares(resid, p0, x_scale="jac", loss="soft_l1", f_scale=2.0, max_nfev=max_iter * 9)
    r = resid(sol.x)
    return np.append(sol.x, 1.0).reshape(3, 3), float(r[:len(line_uv)].mean()), len(line_uv)


def homography_from_corners(img_corners: np.ndarray, spec: PanelSpec) -> np.ndarray:
    return cv2.getPerspectiveTransform(corners(spec).astype(np.float32), img_corners.astype(np.float32)).astype(np.float64)
