"""Offline panel labeller: rough corners on a few anchor frames -> a panel homography for every video frame.

    segkit-panel-label data/panel/<session> --anchors data/panel/<session>/anchors.json

anchors.json: {"<frame>": anchor, ...}, frame numbers as in frames/NNNNN.jpg (1-based, from ffmpeg). An
anchor is either [[x, y] x4]: the cell-area corners TL, TR, BR, BL with cell (0, 0) at TL, or
{"points": [[u, v, x, y], ...]}: four or more panel points (diamonds are easiest: u = 1..cols-1 on the
diamond rows) and where they are in the image, which also works when corners are out of frame.
Rough clicks are snapped onto gridlines and diamonds first. Reviewed point anchors can set "snap": false
to preserve deliberate clicks (especially at low resolution or grazing angles).
An optional "reject": [[first, last], ...] lists frame ranges found wrong in review; they stay unlabelled.

From every anchor the panel is tracked forwards and backwards: KLT corners inside the panel give a
frame-to-frame homography (the panel is flat, so this is exact up to noise). Snapping every step onto the
gridlines was tried and made things worse: at grazing angles the truncated fit happily slides a row onto
its neighbour. So the snap only cleans up the anchors, and the lattice residual is used to score tracks.
A track ends when too few points survive or the panel leaves the frame. Where several tracks cover a
frame, the one with the lower lattice residual wins.

Writes <session>/panel_labels.npz (H [N, 3, 3] panel->image, valid, resid_cells, source anchor) and
review sheets in <session>/review/.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segkit.panel import geometry as G
from segkit.panel.spec import PanelSpec

# Evidence is built on a copy scaled so a cell is about this many pixels: the gridline and diamond filters
# then keep the same size whether the camera is a metre away or touching the glass.
EVIDENCE_CELL_PX = 90.0
TOPHAT_CELLS = 0.17
DIAMOND_CELLS = 0.08
TRUNC_CELLS = 0.08
MAX_SNAP_CELLS = 0.25      # a snap that moves the lattice further than this is distrusted
ANCHOR_MAX_SNAP_CELLS = 0.5
MIN_TRACK_POINTS = 25
MAX_SKIP = 4               # frames a track may jump over (motion blur) before it is declared lost
MIN_VISIBLE_FRAC = 0.04    # of the frame covered by the cell area
KLT = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def frame_paths(session: Path) -> list[Path]:
    return sorted((session / "frames").glob("*.jpg"))


def panel_polygon(H: np.ndarray, spec: PanelSpec) -> np.ndarray:
    return G.project(H, G.corners(spec)).astype(np.float32)


def visible_fraction(H: np.ndarray, spec: PanelSpec, shape: tuple[int, int]) -> float:
    h, w = shape
    poly = panel_polygon(H, spec)
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    try:
        area, _ = cv2.intersectConvexConvex(poly, frame)
    except cv2.error:
        return 0.0
    return float(area) / (w * h)


def visible_cell_px(H: np.ndarray, spec: PanelSpec, shape: tuple[int, int]) -> float:
    """Cell size at the image centre (or the panel centre if the image centre is off the panel)."""
    h, w = shape
    uv = G.unproject(H, np.array([[w / 2, h / 2]]))[0]
    uv = np.clip(uv, [0, 0], [spec.cols - 1, spec.rows - 1])
    return G.cell_px(H, tuple(uv))


def visible_cell_centres(H: np.ndarray, spec: PanelSpec, shape: tuple[int, int]) -> np.ndarray:
    """Panel coordinates of the cell centres that land in the image (image-centre point if none do)."""
    h, w = shape
    uv = np.array([[c + 0.5, r + 0.5] for r in range(spec.rows) for c in range(spec.cols)])
    xy = G.project(H, uv)
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return uv[ok] if ok.any() else G.unproject(H, np.array([[w / 2, h / 2]]))


def _evidence(H: np.ndarray, gray: np.ndarray, spec: PanelSpec):
    """(scale, cell px at that scale, line mask, diamond mask) with filters sized to the visible cells."""
    cell = visible_cell_px(H, spec, gray.shape)
    if not np.isfinite(cell) or cell < 8:
        return None
    s = min(1.0, EVIDENCE_CELL_PX / cell)
    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else gray
    cs = cell * s
    lines = G.line_evidence(small, kernel=max(5, int(TOPHAT_CELLS * cs) | 1))
    dia = G.diamond_evidence(lines, size=max(5, int(DIAMOND_CELLS * cs) | 1))
    return s, cs, lines, dia


def lattice_residual(H: np.ndarray, gray: np.ndarray, spec: PanelSpec) -> float:
    """Mean truncated distance (in cells) from the projected horizontal lines and outer edges to the evidence."""
    ev = _evidence(H, gray, spec)
    if ev is None:
        return float("inf")
    s, cs, lines, _ = ev
    dt = np.minimum(G.distance_map(lines), max(3.0, TRUNC_CELLS * cs))
    xy = G.project(np.diag([s, s, 1.0]) @ H, G.lattice_points(spec, inner_cols=False))
    h, w = dt.shape
    xy = xy[(xy[:, 0] >= 0) & (xy[:, 0] < w - 1) & (xy[:, 1] >= 0) & (xy[:, 1] < h - 1)]
    if len(xy) < 20:
        return float("inf")
    return float(dt[xy[:, 1].astype(int), xy[:, 0].astype(int)].mean() / cs)


def snap(H: np.ndarray, gray: np.ndarray, spec: PanelSpec) -> tuple[np.ndarray, float, bool]:
    """Snap H onto gridline/diamond evidence at a cell-normalised scale. Returns (H, residual in cells, ok)."""
    ev = _evidence(H, gray, spec)
    if ev is None:
        return H, float("inf"), False
    s, cs, lines, dia = ev
    S = np.diag([s, s, 1.0])
    trunc = max(3.0, TRUNC_CELLS * cs)
    Hs, r, n = G.fit_to_lines(S @ H, G.distance_map(lines), spec, G.distance_map(dia), trunc=trunc)
    if n < 20 or not np.isfinite(r):
        return H, float("inf"), False
    Hn = np.linalg.inv(S) @ Hs
    # How far did the snap move the visible part of the lattice, in cells?
    probe = visible_cell_centres(H, spec, gray.shape)
    moved = np.abs(G.unproject(Hn, G.project(H, probe)) - probe).max()
    return Hn, r / cs, bool(moved < MAX_SNAP_CELLS)


def track_step(prev_gray: np.ndarray, gray: np.ndarray, H: np.ndarray, spec: PanelSpec) -> np.ndarray | None:
    """Homography of the panel from prev to this frame, from KLT corners inside the panel."""
    mask = np.zeros(prev_gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, panel_polygon(H, spec).astype(np.int32), 255)
    pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.005, minDistance=8, mask=mask)
    if pts is None or len(pts) < MIN_TRACK_POINTS:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts, None, **KLT)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **KLT)
    good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts, axis=2)[:, 0] < 1.0)
    if good.sum() < MIN_TRACK_POINTS:
        return None
    M, inl = cv2.findHomography(pts[good], nxt[good], cv2.RANSAC, 2.0)
    if M is None or inl.sum() < MIN_TRACK_POINTS:
        return None
    return M @ H


def run_track(paths: list[Path], start: int, H0: np.ndarray, step: int, stop: set[int], spec: PanelSpec,
              grays: dict) -> dict[int, tuple[np.ndarray, float]]:
    """Track from frame index `start` in direction `step` until lost or another anchor. {index: (H, resid)}."""
    out = {}
    H = H0
    i = start
    load = lambda j: grays.setdefault(j, cv2.cvtColor(cv2.imread(str(paths[j])), cv2.COLOR_BGR2GRAY))
    prev = load(i)
    while True:
        # A blurred frame or two breaks KLT; match past them to the next sharp one instead of giving up.
        for skip in range(1, MAX_SKIP + 2):
            j = i + skip * step
            if not 0 <= j < len(paths) or j in stop:
                return out
            g = load(j)
            Ht = track_step(prev, g, H, spec)
            if Ht is not None and visible_fraction(Ht, spec, g.shape) >= MIN_VISIBLE_FRAC:
                break
        else:
            return out
        i, H, prev = j, Ht, g
        out[i] = (H, lattice_residual(H, g, spec))


def review_sheets(session: Path, paths: list[Path], Hs: np.ndarray, valid: np.ndarray, spec: PanelSpec,
                  every: int = 20, per_sheet: int = 24) -> None:
    """Clean crop beside a thin grid, preserving aspect ratio and visible cell details."""
    out = session / "review"
    out.mkdir(exist_ok=True)
    idx = list(range(0, len(paths), every))
    for k in range(0, len(idx), per_sheet):
        tiles = []
        for i in idx[k:k + per_sheet]:
            raw = cv2.imread(str(paths[i]))
            scale = 192 / raw.shape[1]
            im = cv2.resize(raw, (192, round(raw.shape[0] * scale)))
            overlay = im.copy()
            if valid[i]:
                H = np.diag([scale, scale, 1.]) @ Hs[i]
                segments = [((u, 0), (u, spec.rows)) for u in range(spec.cols + 1)]
                segments += [((0, v), (spec.cols, v)) for v in range(spec.rows + 1)]
                for segment in segments:
                    pts = np.round(G.project(H, np.array(segment))).astype(int)
                    cv2.line(overlay, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 1)
            tile = cv2.copyMakeBorder(np.hstack([im, overlay]), 24, 0, 0, 0, cv2.BORDER_CONSTANT)
            text = f"{paths[i].stem} anchor {i + 1} " + ("VALID" if valid[i] else "excluded")
            cv2.putText(tile, text, (4, 17), 0, .45, (255, 255, 255), 1)
            tiles.append(tile)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(out / f"sheet_{k // per_sheet:02d}.jpg"),
                    np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))


def draw_cells(im: np.ndarray, H: np.ndarray, spec: PanelSpec, colour=(0, 0, 255)) -> None:
    """Gridlines plus each cell's (row, col) label at its centre."""
    h, w = im.shape[:2]
    for u in range(spec.cols + 1):
        a, b = G.project(H, np.array([[u, 0], [u, spec.rows]]))
        cv2.line(im, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, 3)
    for v in range(spec.rows + 1):
        a, b = G.project(H, np.array([[0, v], [spec.cols, v]]))
        cv2.line(im, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), colour, 3)
    for r in range(spec.rows):
        for c in range(spec.cols):
            x, y = G.project(H, np.array([[c + 0.5, r + 0.5]]))[0]
            if 0 <= x < w and 0 <= y < h:
                size = np.clip(G.cell_px(H, (c, r)) / 90, 0.6, 4)
                cv2.putText(im, f"{r},{c}", (int(x) - int(30 * size), int(y) + int(10 * size)), 0, size,
                            (0, 255, 255), max(2, int(2 * size)))


def anchor_homography(anchor, spec: PanelSpec) -> np.ndarray:
    if isinstance(anchor, dict):
        p = np.array(anchor["points"], float)
        H, _ = cv2.findHomography(p[:, :2], p[:, 2:], 0)
        return H
    return G.homography_from_corners(np.array(anchor, float), spec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--anchors", type=Path)
    ap.add_argument("--crop-input", action="store_true", help="frames are already phone analysis crops")
    ap.add_argument("--review-every", type=int, default=20, help="use 1 to review every frame")
    ap.add_argument("--spec", type=Path, help="panel profile JSON (default: built-in)")
    args = ap.parse_args()
    spec = PanelSpec.load(args.spec)
    paths = frame_paths(args.session)
    anchors = json.loads((args.anchors or args.session / "anchors.json").read_text())
    rejects = anchors.pop("reject", [])
    anchor_idx = {int(f) - 1: anchor_homography(c, spec) for f, c in anchors.items()}

    best: dict[int, tuple[np.ndarray, float, int]] = {}

    def offer(i, H, r, src):
        if i not in best or r < best[i][1]:
            best[i] = (H, r, src)

    grays: dict[int, np.ndarray] = {}
    for a, quad in tqdm(sorted(anchor_idx.items()), desc="anchors"):
        g = grays.setdefault(a, cv2.cvtColor(cv2.imread(str(paths[a])), cv2.COLOR_BGR2GRAY))
        H0 = quad
        anchor = anchors[str(a + 1)]
        if isinstance(anchor, dict) and anchor.get("snap") is False:
            H, r = H0, lattice_residual(H0, g, spec)
        else:
            H, r, ok = snap(H0, g, spec)
            if not ok:
                print(f"warning: anchor {a + 1} snap rejected; preserving its clicks")
                H, r = H0, lattice_residual(H0, g, spec)
        moved = np.abs(G.unproject(H, G.project(H0, G.corners(spec))) - G.corners(spec)).max()
        if moved > ANCHOR_MAX_SNAP_CELLS:
            print(f"warning: anchor {a + 1} snapped {moved:.2f} cells from its clicks; check its corners")
        offer(a, H, -1.0, a + 1)   # anchors always win their own frame
        stop = set(anchor_idx) - {a}
        for step in (1, -1):
            for i, (Hi, ri) in run_track(paths, a, H, step, stop, spec, grays).items():
                offer(i, Hi, ri, a + 1)
        grays.clear()

    n = len(paths)
    Hs = np.zeros((n, 3, 3))
    valid = np.zeros(n, bool)
    resid = np.full(n, np.inf)
    source = np.zeros(n, int)
    for a, b in rejects:
        for i in range(a - 1, b):
            best.pop(i, None)
    for i, (H, r, src) in best.items():
        Hs[i], valid[i], resid[i], source[i] = H, True, r, src
    np.savez(args.session / "panel_labels.npz", H=Hs, valid=valid, resid_cells=resid, source=source,
             coordinate_space="analysis_crop" if args.crop_input else "video",
             spec=json.dumps(spec.__dict__ | {"diamond_rows": list(spec.diamond_rows)}))
    print(f"{valid.sum()}/{n} frames labelled; resid median {np.median(resid[valid & np.isfinite(resid)]):.3f} cells")
    review_sheets(args.session, paths, Hs, valid, spec, every=args.review_every)


if __name__ == "__main__":
    main()
