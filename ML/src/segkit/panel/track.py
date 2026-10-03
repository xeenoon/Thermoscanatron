"""Panel cell tracker: PanelNet per frame + memory of which cell is which.

    segkit-panel-track data/panel/<session> --model runs/panel_v1/panelseg.pte --video runs/panel_v1/track.mp4

Works on the phone's input: a centred square of each frame (90% of the short side) at the model size. State
is a homography H from panel coordinates (u across columns, v down rows, in cells) to crop pixels.

  acquire  Enough of the panel outline must be in view once (the "step back" moment: a corner, or two
           crossing edges). The phase field is unwrapped across the visible panel into continuous (u, v), up
           to a whole cell in u and two rows in v; a RANSAC homography fits that, and the integer offset
           whose cell area best matches the predicted mask is chosen. The model's upright prior plus the
           diamond row parity settle which end is cell (0, 0).
  follow   KLT on the panel predicts this frame's H. Each panel pixel's predicted within-cell phase is
           unwrapped against that prediction: u = phase_u + nearest integer to (u_pred - phase_u), and v
           likewise on a period of two rows. A RANSAC homography over those pixels is the new H. The model
           pins the position inside a cell every frame, so nothing drifts; KLT only has to be right to
           within half a cell (two rows for v) for the integer part to carry over.
  correct  Off-by-one candidates (u +- 1, v +- 2) are scored against the predicted panel mask; a visible
           panel edge or corner then fixes any slip. With no edge in view all candidates tie and H stays.
  lost     After LOST_AFTER frames without a good fit the panel is dropped until it is acquired again.

With labels (panel_labels.npz), scores the cell under the crop centre against them.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD
from segkit.datasets.panels import is_val, load_session, targets
from segkit.panel import geometry as G
from segkit.panel.spec import PanelSpec

BOX_FRACTION = 0.9
SAMPLE_STRIDE = 4
MIN_FIT_POINTS = 150
MIN_INLIER_FRAC = 0.5
RANSAC_PX = 4.0           # inlier threshold floor (far away)
RANSAC_CELLS = 0.06       # ...but at least this fraction of a cell: up close a cell is hundreds of px wide
LOST_AFTER = 8
LOST_AFTER_ON_PANEL = 60  # while the panel fills most of the view we are still on it: coast much longer
ON_PANEL_FRACTION = 0.6
SHIFT_MARGIN = 0.04       # IoU gain an off-by-one candidate needs before it replaces the current cells
ACQUIRE_MIN_AREA = 0.05   # of the crop covered by panel before acquiring is tried
ACQUIRE_MIN_IOU = 0.6
ACQUIRE_MARGIN = 0.08     # the winning cell offset must beat the runner-up by this much IoU
BM_SCALE = 4
BM_RADIUS = 8             # search +-8 px at 1/4 scale = +-32 crop px per frame
KLT = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


class Model:
    """PanelNet deployment wrapper, from a .pte (as on the phone) or a .pt checkpoint."""

    def __init__(self, path: Path, size: int):
        self.size = size
        if path.suffix == ".pte":
            from executorch.runtime import Runtime
            self.method = Runtime.get().load_program(path).load_method("forward")
            self.run = lambda x: self.method.execute([x])
        else:
            from segkit.models.panelnet import PanelNet, PanelProbabilityHead
            net = PanelNet()
            net.load_state_dict(torch.load(path, map_location="cpu"))
            head = PanelProbabilityHead(net).eval()
            self.run = lambda x: head(x)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        x = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        dense, present = self.run(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None])
        return dense[0].numpy(), float(present[0, 0])


@dataclass
class Result:
    H: np.ndarray | None
    state: str
    centre_cell: tuple[int, int] | None


def translate(du: float, dv: float) -> np.ndarray:
    return np.array([[1, 0, du], [0, 1, dv], [0, 0, 1]], np.float64)


class PanelTracker:
    def __init__(self, spec: PanelSpec, size: int, klt: bool = True, block_match: bool = False):
        self.spec, self.size, self.use_klt, self.use_bm = spec, size, klt, block_match
        self.H: np.ndarray | None = None
        self.last_frac: float | None = None
        self.misses = 0
        self.prev_gray: np.ndarray | None = None

    # -- model output -> samples -------------------------------------------------------------------------
    def samples(self, dense: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixels on the panel (stride SAMPLE_STRIDE): xy, phase_u in [0, 1), phase_v in [0, 2)."""
        s = SAMPLE_STRIDE
        m = dense[0, s // 2::s, s // 2::s] > 0.5
        ys, xs = np.nonzero(m)
        xy = np.c_[xs * s + s // 2 + 0.5, ys * s + s // 2 + 0.5]
        d = dense[:, s // 2::s, s // 2::s][:, ys, xs]
        pu = np.arctan2(d[2], d[3]) / (2 * math.pi) % 1.0
        pv = np.arctan2(d[4], d[5]) / math.pi % 2.0
        return xy, pu, pv

    def fit(self, H_pred: np.ndarray, xy, pu, pv) -> tuple[np.ndarray | None, float]:
        """Unwrap the phases against H_pred, fit H. Returns (H, inlier fraction)."""
        if len(xy) < MIN_FIT_POINTS:
            return None, 0.0
        uv_pred = G.unproject(H_pred, xy)
        u = pu + np.round(uv_pred[:, 0] - pu)
        v = pv + 2 * np.round((uv_pred[:, 1] - pv) / 2)
        H, inl = cv2.findHomography(np.c_[u, v], xy, cv2.RANSAC, ransac_px(np.c_[u, v], xy), maxIters=500)
        if H is None:
            return None, 0.0
        return H, float(inl.mean())

    def mask_iou(self, H: np.ndarray, mask: np.ndarray) -> float:
        poly = np.zeros_like(mask, np.uint8)
        q = G.project(H, G.corners(self.spec))
        if not np.isfinite(q).all() or np.abs(q).max() > 1e5:
            return 0.0
        cv2.fillConvexPoly(poly, np.round(q).astype(np.int32), 1)
        inter = (poly & mask).sum()
        union = (poly | mask).sum()
        return inter / union if union else 0.0

    def offset_ious(self, H: np.ndarray, dense: np.ndarray) -> list[tuple[float, int, int]]:
        """IoU of the cell area with the predicted mask for every whole-cell renumbering (u + du, v + dv, dv even)
        of H, on a 1/4-resolution grid: shifting the numbering just adds integers to each pixel's (u, v)."""
        q = 4
        mask = dense[0, q // 2::q, q // 2::q] > 0.5
        ys, xs = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
        xy = np.c_[xs.ravel() * q + q / 2, ys.ravel() * q + q / 2]
        p = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(H).T
        ok = p[:, 2] > 0 if np.linalg.det(H) > 0 else p[:, 2] < 0
        u, v = p[:, 0] / p[:, 2], p[:, 1] / p[:, 2]
        m = mask.ravel()
        out = []
        for du in range(-self.spec.cols, self.spec.cols + 1):
            for dv in range(-self.spec.rows - 1, self.spec.rows + 2, 2):
                inside = ok & (u + du >= 0) & (u + du <= self.spec.cols) & (v + dv >= 0) & (v + dv <= self.spec.rows)
                union = (inside | m).sum()
                out.append(((inside & m).sum() / union if union else 0.0, du, dv))
        return out

    def correct(self, H: np.ndarray, dense: np.ndarray) -> np.ndarray:
        """Renumber the cells if a visible panel edge or corner says so: the whole-cell offset whose cell area
        matches the predicted mask clearly better than the current numbering wins. With no edge in view all
        offsets tie and the numbering carried over from earlier frames stays."""
        scores = self.offset_ious(H, dense)
        current = next(s for s, du, dv in scores if du == 0 and dv == 0)
        best, du, dv = max(scores)
        if best > current + SHIFT_MARGIN:
            return H @ translate(-du, -dv)
        return H

    def acquire(self, dense: np.ndarray) -> np.ndarray | None:
        """Find H with no prior: unwrap the phase field across the visible panel, fit H (known up to a whole
        cell in u and two rows in v), then pick the integer offset whose cell area matches the predicted mask.
        Needs enough of the outline in view (a corner, or two crossing edges) for one offset to win clearly."""
        s = SAMPLE_STRIDE
        g = dense[:, s // 2::s, s // 2::s]
        mask = g[0] > 0.5
        if mask.mean() < ACQUIRE_MIN_AREA:
            return None
        u, v = unwrap(mask, np.arctan2(g[2], g[3]) / (2 * math.pi), np.arctan2(g[4], g[5]) / math.pi)
        ok = np.isfinite(u)
        if ok.sum() < MIN_FIT_POINTS:
            return None
        ys, xs = np.nonzero(ok)
        xy = np.c_[xs * s + s // 2 + 0.5, ys * s + s // 2 + 0.5]
        uv = np.c_[u[ok], v[ok]]
        H, inl = cv2.findHomography(uv, xy, cv2.RANSAC, ransac_px(uv, xy), maxIters=1000)
        if H is None or inl.mean() < MIN_INLIER_FRAC:
            return None
        scores = sorted(self.offset_ious(H, dense), reverse=True)
        (best, du, dv), (second, _, _) = scores[0], scores[1]
        if best < ACQUIRE_MIN_IOU or best - second < ACQUIRE_MARGIN:
            return None
        return H @ translate(-du, -dv)

    def unwrapped_fit(self, dense: np.ndarray) -> np.ndarray | None:
        """H from the phase field alone (spatially unwrapped), right up to a whole cell in u and two rows in v."""
        s = SAMPLE_STRIDE
        g = dense[:, s // 2::s, s // 2::s]
        mask = g[0] > 0.5
        if mask.sum() < MIN_FIT_POINTS:
            return None
        u, v = unwrap(mask, np.arctan2(g[2], g[3]) / (2 * math.pi), np.arctan2(g[4], g[5]) / math.pi)
        ok = np.isfinite(u)
        if ok.sum() < MIN_FIT_POINTS:
            return None
        ys, xs = np.nonzero(ok)
        xy = np.c_[xs * s + s // 2 + 0.5, ys * s + s // 2 + 0.5]
        uv = np.c_[u[ok], v[ok]]
        H, inl = cv2.findHomography(uv, xy, cv2.RANSAC, ransac_px(uv, xy), maxIters=1000)
        if H is None or inl.mean() < MIN_INLIER_FRAC:
            return None
        return H

    def relock(self, dense: np.ndarray, H_pred: np.ndarray) -> np.ndarray | None:
        """When the predicted grid has gone stale (e.g. moving in fast), rebuild it from this frame's phase and
        take the integer cell offset that puts the crop centre where the prediction had it. Moving in scales
        about the centre, so the centre cell is the part of a stale prediction that is still right."""
        H = self.unwrapped_fit(dense)
        if H is None:
            return None
        c = np.array([[self.size / 2, self.size / 2]])
        want = G.unproject(H_pred, c)[0]
        have = G.unproject(H, c)[0]
        du = np.round(want[0] - have[0])
        dv = 2 * np.round((want[1] - have[1]) / 2)
        return H @ translate(-du, -dv)

    def step(self, crop_rgb: np.ndarray, dense: np.ndarray, present: float) -> Result:
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        xy, pu, pv = self.samples(dense)
        state = "lost"
        self.last_frac = None
        if self.H is not None:
            if self.prev_gray is not None and self.use_klt:
                H_pred = self.klt(gray)
            elif self.prev_gray is not None and self.use_bm:
                dx, dy, _ = block_shift(self.prev_gray, gray)
                H_pred = translate(dx, dy) @ self.H
            else:
                H_pred = self.H
            H, frac = self.fit(H_pred, xy, pu, pv) if present > 0.5 else (None, 0.0)
            self.last_frac = frac
            if H is not None and frac >= MIN_INLIER_FRAC:
                self.H = self.correct(H, dense)
                self.misses = 0
                state = "tracking"
            elif present > 0.5 and (H := self.relock(dense, H_pred)) is not None:
                self.H = self.correct(H, dense)
                self.misses = 0
                state = "relocked"
            else:
                # Coast (blur, glare); while the panel still fills the view we are on it, so wait much longer.
                self.H = H_pred
                self.misses += 1
                state = "coasting"
                on_panel = present > 0.5 and (dense[0] > 0.5).mean() > ON_PANEL_FRACTION
                if self.misses > (LOST_AFTER_ON_PANEL if on_panel else LOST_AFTER):
                    self.H, state = None, "lost"
        if self.H is None and present > 0.5:
            H = self.acquire(dense)
            if H is not None:
                self.H, self.misses, state = H, 0, "acquired"
        self.prev_gray = gray
        return Result(self.H, state, self.centre_cell())

    def coast(self, crop_rgb: np.ndarray) -> np.ndarray | None:
        """A frame without a model output: carry H along the measured image motion only."""
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        if self.H is not None and self.prev_gray is not None:
            dx, dy, _ = block_shift(self.prev_gray, gray)
            self.H = translate(dx, dy) @ self.H
        self.prev_gray = gray
        return self.H

    def klt(self, gray: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(gray)
        q = G.project(self.H, G.corners(self.spec))
        if np.isfinite(q).all() and np.abs(q).max() < 1e5:
            cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 255)
        pts = cv2.goodFeaturesToTrack(self.prev_gray, 200, 0.01, 6, mask=mask)
        if pts is None or len(pts) < 15:
            return self.H
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **KLT)
        ok = st[:, 0] == 1
        if ok.sum() < 15:
            return self.H
        M, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 3.0)
        return self.H if M is None else M @ self.H

    def centre_cell(self) -> tuple[int, int] | None:
        if self.H is None:
            return None
        return cell_at(self.H, self.spec, self.size / 2, self.size / 2)


def unwrap(mask: np.ndarray, pu: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Breadth-first phase unwrapping over the mask from its most interior pixel. pu has period 1, pv period 2.
    Steps across a gridline wrap correctly as long as neighbouring samples are less than half a period apart.
    Returns u, v with NaN off the mask (or off the seed's connected region)."""
    h, w = mask.shape
    u = np.full((h, w), np.nan)
    v = np.full((h, w), np.nan)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    sy, sx = np.unravel_index(np.argmax(dist), dist.shape)
    u[sy, sx], v[sy, sx] = pu[sy, sx] % 1, pv[sy, sx] % 2
    queue = [(sy, sx)]
    head = 0
    while head < len(queue):
        y, x = queue[head]
        head += 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and np.isnan(u[ny, nx]):
                du = (pu[ny, nx] - pu[y, x] + 0.5) % 1 - 0.5
                dv = (pv[ny, nx] - pv[y, x] + 1) % 2 - 1
                u[ny, nx] = u[y, x] + du
                v[ny, nx] = v[y, x] + dv
                queue.append((ny, nx))
    return u, v


def ransac_px(uv: np.ndarray, xy: np.ndarray) -> float:
    """Inlier threshold: RANSAC_CELLS of the local cell size (affine least-squares fit of uv -> xy), >= RANSAC_PX."""
    A = np.c_[uv, np.ones(len(uv))]
    sol, *_ = np.linalg.lstsq(A, xy, rcond=None)
    cell = math.sqrt(abs(np.linalg.det(sol[:2].T)))
    return max(RANSAC_PX, RANSAC_CELLS * cell)


def block_shift(prev: np.ndarray, cur: np.ndarray, radius: int = BM_RADIUS) -> tuple[float, float, float]:
    """Image translation from prev to cur (grey crops), by exhaustive SAD block matching on a 1/4-scale copy.
    Returns (dx, dy) in crop pixels and the match quality (best SAD vs the median: lower is more distinct).
    Cheap enough for the phone (no OpenCV there), and up close the panel's dirt and scratches give it texture."""
    q = BM_SCALE
    a = cv2.resize(prev, None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(cur, None, fx=1 / q, fy=1 / q, interpolation=cv2.INTER_AREA).astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    n = a.shape[0]
    m = radius
    win = a[m:n - m, m:n - m]
    best, bx, by, costs = np.inf, 0, 0, []
    for dy in range(-m, m + 1):
        for dx in range(-m, m + 1):
            c = float(np.abs(b[m + dy:n - m + dy, m + dx:n - m + dx] - win).mean())
            costs.append(c)
            if c < best:
                best, bx, by = c, dx, dy
    return bx * q, by * q, best / max(np.median(costs), 1e-6)


def cell_at(H: np.ndarray, spec: PanelSpec, x: float, y: float) -> tuple[int, int] | None:
    u, v = G.unproject(H, np.array([[x, y]]))[0]
    if 0 <= u < spec.cols and 0 <= v < spec.rows:
        return int(v), int(u)
    return None


def centre_crop(frame_rgb: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Phone-style analysis crop. Returns (crop, 3x3 frame->crop)."""
    h, w = frame_rgb.shape[:2]
    side = BOX_FRACTION * min(h, w)
    x0, y0 = (w - side) / 2, (h - side) / 2
    s = size / side
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s], [0, 0, 1]])
    return cv2.warpAffine(frame_rgb, A[:2], (size, size), flags=cv2.INTER_AREA), A


def draw(crop_bgr: np.ndarray, res: Result, spec: PanelSpec, dense: np.ndarray, label: tuple | None) -> np.ndarray:
    from segkit.panel.label import draw_cells
    out = crop_bgr.copy()
    tint = np.zeros_like(out)
    tint[dense[0] > 0.5] = (0, 90, 0)
    out = cv2.addWeighted(out, 1.0, tint, 0.6, 0)
    if res.H is not None:
        draw_cells(out, res.H, spec, colour=(0, 0, 255) if res.state != "coasting" else (0, 165, 255))
    c = out.shape[0] // 2
    cv2.drawMarker(out, (c, c), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    txt = f"{res.state}"
    if res.centre_cell:
        txt += f"  cell {res.centre_cell[0]},{res.centre_cell[1]}"
    cv2.putText(out, txt, (8, 26), 0, 0.7, (255, 255, 255), 2)
    if label is not None:
        cv2.putText(out, f"label {label[0]},{label[1]}" if label != () else "label -", (8, 52), 0, 0.6,
                    (200, 200, 200), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-panel-track", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--model", type=Path, help=".pte or .pt checkpoint")
    ap.add_argument("--oracle", action="store_true",
                    help="feed the tracker the label targets instead of the model (tests the tracker alone)")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--video", type=Path, help="write an overlay video here")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th frame (the phone runs the model at ~8 fps)")
    ap.add_argument("--no-klt", action="store_true", help="no optical-flow motion prior (as on the phone)")
    ap.add_argument("--block-match", action="store_true", help="block-matching motion prior (what the phone runs)")
    ap.add_argument("--crops", action="store_true",
                    help="run on the session's recorded analysis crops (exactly the phone's model input) instead "
                         "of video frames; no labels, prints the tracker's state timeline")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    if args.crops:
        return run_crops(args)
    frames, Hs, valid, spec = load_session(args.session)
    model = None if args.oracle else Model(args.model, args.size)
    tracker = PanelTracker(spec, args.size, klt=not args.no_klt, block_match=args.block_match)
    end = args.end or len(frames)
    writer = None
    if args.video:
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (args.size, args.size))

    rows = []
    H_video = np.full((len(frames), 3, 3), np.nan)
    for i in tqdm(range(args.start - 1, end, args.stride)):
        rgb = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)
        crop, A = centre_crop(rgb, args.size)
        if model is None:
            dense = targets(A @ Hs[i], args.size, spec) if valid[i] else np.zeros((6, args.size, args.size))
            present = float(dense[0].mean() > 0.02)
        else:
            dense, present = model(crop)
        res = tracker.step(crop, dense, present)
        label = None
        if valid[i]:
            label = cell_at(A @ Hs[i], spec, args.size / 2, args.size / 2) or ()
        rows.append((i, res.state, res.centre_cell, label))
        if res.H is not None:
            H_video[i] = np.linalg.inv(A) @ res.H
        if writer:
            writer.write(draw(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), res, spec, dense, label))
    if writer:
        writer.release()

    def report(name, rs):
        scored = [r for r in rs if r[3] not in (None, ())]          # labelled, centre on the panel
        said = [r for r in scored if r[2] is not None]
        right = [r for r in said if r[2] == r[3]]
        off_panel = [r for r in rs if r[3] == () and r[2] is not None]
        print(f"{name}: centre on panel in {len(scored)} labelled frames; tracker gives a cell in "
              f"{len(said)} ({len(said) / max(1, len(scored)):.0%}), correct {len(right)}/{len(said)} "
              f"({len(right) / max(1, len(said)):.1%}); claims a cell off the panel in {len(off_panel)}")
    report("all", rows)
    report("val blocks", [r for r in rows if is_val(r[0])])
    out = args.session / ("track_oracle.npz" if args.oracle else "track.npz")
    np.savez(out, H=H_video, state=np.array([r[1] for r in rows] + [""] * (len(frames) - len(rows)))[:len(frames)],
             first=args.start - 1)
    print(f"tracker homographies (video pixels) -> {out}")


def crop_labels(session: Path, n: int, size: int) -> list | None:
    """Label homographies (video frames) mapped onto the recorded analysis crops, nearest video frame in time."""
    if not (session / "panel_labels.npz").exists():
        return None
    import csv
    import json
    from segkit.panel.thermal import video_geometry, video_times
    frames, Hs, valid, _ = load_session(session)
    _, s, ox = video_geometry(session)
    vt = video_times(session, len(frames))
    an = json.loads((session / "meta.json").read_text())["analysis"]
    k = size / an["box_side"]
    A = np.array([[s * k, 0, (ox - an["box_left"]) * k], [0, s * k, -an["box_top"] * k], [0, 0, 1]])
    ts = [int(r["sensor_ts_ns"]) for r in csv.DictReader((session / "frames.csv").open())]
    out = []
    for i in range(n):
        j = int(np.argmin(np.abs(vt - ts[i])))
        out.append(A @ Hs[j] if valid[j] and abs(vt[j] - ts[i]) < 20e6 else None)
    return out


def run_crops(args) -> None:
    """Replay the phone's own model input (session crops/, one per analysed camera frame) through the tracker."""
    crops = sorted((args.session / "crops").glob("*.jpg"))
    spec = PanelSpec()
    model = Model(args.model, args.size)
    tracker = PanelTracker(spec, args.size, klt=not args.no_klt, block_match=args.block_match)
    labels = crop_labels(args.session, len(crops), args.size)
    writer = None
    if args.video:
        writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (args.size, args.size))
    log, scored, fits = [], [], []
    for i in tqdm(range(args.start - 1, args.end or len(crops), args.stride)):
        crop = cv2.cvtColor(cv2.imread(str(crops[i])), cv2.COLOR_BGR2RGB)
        dense, present = model(crop)
        res = tracker.step(crop, dense, present)
        log.append((i, res.state, res.centre_cell, present, float((dense[0] > 0.5).mean())))
        fits.append(tracker.last_frac)
        if labels is not None and labels[i] is not None:
            scored.append((res.centre_cell, cell_at(labels[i], spec, args.size / 2, args.size / 2)))
        if writer:
            out = draw(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), res, spec, dense, None)
            cv2.putText(out, str(i), (330, 370), 0, 0.6, (255, 255, 255), 1)
            writer.write(out)
    if writer:
        writer.release()
    # Timeline: runs of the same state.
    runs = []
    for i, st, cell, pr, area in log:
        if runs and runs[-1][1] == st:
            runs[-1][2] = i
        else:
            runs.append([i, st, i])
    print(" ".join(f"{a}-{b}:{st}" for a, st, b in runs))
    states = [st for _, st, _, _, _ in log]
    print("states: " + ", ".join(f"{k} {states.count(k)}" for k in sorted(set(states))) +
          f"; mean phase-fit inlier fraction {np.mean([f for f in fits if f is not None] or [0]):.2f}")
    on = [(c, l) for c, l in scored if l is not None]
    said = [(c, l) for c, l in on if c is not None]
    print(f"labelled crops with the centre on the panel: {len(on)}; tracker gives a cell in {len(said)} "
          f"({len(said) / max(1, len(on)):.0%}), correct {sum(c == l for c, l in said)}/{len(said)} "
          f"({sum(c == l for c, l in said) / max(1, len(said)):.1%})")
    np.save(args.session / "track_crops.npy", np.array([(i, st, str(c), p, a) for i, st, c, p, a in log], dtype=object))


if __name__ == "__main__":
    main()
