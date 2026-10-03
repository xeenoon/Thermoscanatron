"""Per-cell panel temperatures and hotspot labels from the thermal stream.

    segkit-panel-thermal data/panel/<session> --calib data/panel/thermal_calib.json [--source labels|tracker]

For every frame with a panel homography (labeller output, or the live tracker's):
  1. Panel pose in 3D: H = K [r1 r2 t] diag(cell_w, cell_h) with K the phone camera's intrinsics in video
     pixels; the profile's cell size sets the scale.
  2. The thermal packet for that frame: arrival time = camera capture time + the calibration's latency.
  3. Each thermal pixel's footprint (3 x 3 rays over +-0.75 px, i.e. the pixel plus the optics' blur) is cast
     from the thermal camera (pose from segkit.thermal_calib) onto the panel plane. A pixel whose whole
     footprint lands inside one cell is a clean reading of that cell; one whose footprint is inside the cell
     area (but may straddle cells) counts towards the panel average. Mixed pixels (frame, background)
     are dropped.
  4. Cell temperature = median of its clean pixels in that frame; panel average = mean of all panel pixels.
Hotspot: a cell whose temperature differs from the panel average by more than HOTSPOT_DELTA_C (either way).

Session summary: per cell, the median of its per-frame temperatures relative to the same frame's panel average
(so the panel warming or the sky changing during the walk cancels out), plus the median panel average.

Writes <session>/thermal_cells.csv (one row per frame), thermal_summary.json, thermal_review/*.jpg.
Readings are apparent temperatures at emissivity 0.95: glass also reflects the sky, so a cold sky reads low.
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segkit.datasets.panels import load_session
from segkit.panel import geometry as G
from segkit.panel.spec import PanelSpec

HOTSPOT_DELTA_C = 5.0
TW, TH = 32, 24
RECORD_BYTES = 8 + 1566
FOOTPRINT = np.array([(dx, dy) for dy in (-0.75, 0, 0.75) for dx in (-0.75, 0, 0.75)])
EDGE_MARGIN_CELLS = 0.05      # keep footprints this far inside the cell area (frame and gaps are not cell)
MAX_PACKET_DT_MS = 100        # no thermal packet this close to the wanted time: frame skipped
MIN_PANEL_PX = 6              # thermal pixels on the panel before a frame's panel average is trusted


# ---------------------------------------------------------------------------------------------- inputs

def thermal_stream(session: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = (session / "thermal.bin").read_bytes()
    n = len(raw) // RECORD_BYTES
    t = np.array([struct.unpack_from("<q", raw, i * RECORD_BYTES)[0] for i in range(n)])
    temps = np.stack([np.frombuffer(raw, "<i2", TW * TH, i * RECORD_BYTES + 8 + 28).reshape(TH, TW) / 100
                      for i in range(n)])
    return t, temps


def video_geometry(session: Path) -> tuple[np.ndarray, float, float]:
    """Camera intrinsics in video pixels, and (scale, x offset) mapping video -> analysis frame pixels.

    Both streams come from the same sensor: the analysis frame is the full 4:3 sensor, the video a 16:9 crop of
    it across the short side (checked by matching the recorded analysis crops: correlation 0.99)."""
    meta = json.loads((session / "meta.json").read_text())
    cam, an = meta["camera"], meta["analysis"]
    vw, vh = cv2.imread(str(next((session / "frames").glob("*.jpg")))).shape[1::-1]
    s = an["upright_h"] / vh
    ox = (an["upright_w"] - vw * s) / 2
    f_an = cam["focal_lengths_mm"][0] / cam["sensor_physical_mm"][0] * cam["pixel_array"][0] * an["sensor_to_buffer"][0]
    f = f_an / s
    K = np.array([[f, 0, (an["upright_w"] / 2 - ox) / s], [0, f, an["upright_h"] / 2 / s], [0, 0, 1]])
    return K, s, ox


def video_times(session: Path, n_frames: int, search: range = range(-30, 31)) -> np.ndarray:
    """Capture time (camera sensor clock, ns) of each video frame.

    The video starts a few frames after the analysis stream. The offset is found by matching recorded analysis
    crops against video frames (same geometry as video_geometry) over a range of frame offsets."""
    rows = list(csv.DictReader((session / "frames.csv").open()))
    ts = np.array([int(r["sensor_ts_ns"]) for r in rows])
    meta = json.loads((session / "meta.json").read_text())
    an = meta["analysis"]
    _, s, ox = video_geometry(session)
    fps = 30000 / 1001
    frames = sorted((session / "frames").glob("*.jpg"))
    k_px = an["model_size"] / an["box_side"]
    A = np.array([[s * k_px, 0, (ox - an["box_left"]) * k_px], [0, s * k_px, -an["box_top"] * k_px]], np.float32)
    size = an["model_size"]
    picks = np.linspace(len(rows) * 0.1, len(rows) * 0.9, 6).astype(int)
    score = {}
    cache = {}
    for k in picks:
        crop = cv2.imread(str(session / "crops" / f"{k:06d}.jpg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        base = (ts[k] - ts[0]) / 1e9 * fps
        for off in search:
            j = int(round(base + off))
            if not 0 <= j < n_frames:
                continue
            if j not in cache:
                cache[j] = cv2.warpAffine(cv2.imread(str(frames[j]), cv2.IMREAD_GRAYSCALE), A, (size, size))
            w = cache[j]
            ok = w > 0
            score[off] = score.get(off, 0.0) + float(np.corrcoef(w[ok], crop[ok])[0, 1])
    best = max(score, key=score.get)
    # Frame j shows sensor time ts0 + (j - best) / fps.
    return ts[0] + (np.arange(n_frames) - best) / fps * 1e9


def load_calib(path: Path) -> dict:
    c = json.loads(path.read_text())
    R = rotation(*np.radians([c["yaw_deg"], c["pitch_deg"], c["roll_deg"]]))
    return {**c, "R": R, "c_m": np.array([c["x_cm"], c["y_cm"], c["z_cm"]]) / 100}


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Same convention as segkit.thermal_calib: thermal axes in camera coordinates, Ry @ Rx @ Rz."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def footprint_rays(calib: dict) -> np.ndarray:
    """(TH, TW, 9, 3) unit rays in camera coordinates: each thermal pixel's footprint (equidistant lens)."""
    v, u = np.mgrid[0:TH, 0:TW]
    uv = np.stack([u, v], -1)[:, :, None, :] + FOOTPRINT[None, None]
    mx = (uv[..., 0] - calib["thermal_cx"]) / calib["thermal_fx"] * (-1 if calib["mirror"] else 1)
    my = (uv[..., 1] - calib["thermal_cy"]) / calib["thermal_fy"]
    th = np.hypot(mx, my)
    s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
    d = np.stack([mx * s, my * s, np.cos(th)], -1)
    return d @ calib["R"].T


# ---------------------------------------------------------------------------------------------- geometry

def panel_pose(H: np.ndarray, K: np.ndarray, spec: PanelSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(origin t, unit column axis r1, unit row axis r2) of the panel in camera coordinates, metres."""
    B = np.linalg.inv(K) @ H
    lam = np.linalg.norm(B[:, 0]) / spec.cell_w_m
    if B[2, 2] < 0:
        B = -B
    r1 = B[:, 0] / np.linalg.norm(B[:, 0])
    r2 = B[:, 1] / np.linalg.norm(B[:, 1])
    return B[:, 2] / lam, r1, r2


def cast(rays: np.ndarray, calib: dict, t: np.ndarray, r1: np.ndarray, r2: np.ndarray,
         spec: PanelSpec) -> np.ndarray:
    """Panel (u, v) in cells where each ray from the thermal camera meets the panel plane (NaN if behind)."""
    n = np.cross(r1, r2)
    c = calib["c_m"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((t - c) @ n) / (rays @ n)
    p = c + s[..., None] * rays - t
    uv = np.stack([p @ r1 / spec.cell_w_m, p @ r2 / spec.cell_h_m], -1)
    uv[s <= 0] = np.nan
    return uv


def frame_readings(uv: np.ndarray, temps: np.ndarray, spec: PanelSpec):
    """-> (cell temps [rows, cols] NaN if unseen, clean pixel counts, panel average, panel pixel count,
    per-pixel cell index or -1, per-pixel on-panel flag)."""
    m = EDGE_MARGIN_CELLS
    u, v = uv[..., 0], uv[..., 1]
    ok = np.isfinite(u) & np.isfinite(v)
    on_panel = (ok & (u >= m) & (u <= spec.cols - m) & (v >= m) & (v <= spec.rows - m)).all(-1)
    cu, cv = np.floor(np.nan_to_num(u)), np.floor(np.nan_to_num(v))
    fu, fv = np.nan_to_num(u) - cu, np.nan_to_num(v) - cv
    inner = (fu >= m) & (fu <= 1 - m) & (fv >= m) & (fv <= 1 - m)
    same = (cu == cu[..., :1]).all(-1) & (cv == cv[..., :1]).all(-1) & inner.all(-1)
    clean = on_panel & same
    cell_idx = np.where(clean, cv[..., 0] * spec.cols + cu[..., 0], -1).astype(int)
    cell_t = np.full((spec.rows, spec.cols), np.nan)
    count = np.zeros((spec.rows, spec.cols), int)
    for k in np.unique(cell_idx[cell_idx >= 0]):
        vals = temps[cell_idx == k]
        cell_t.flat[k] = np.median(vals)
        count.flat[k] = len(vals)
    panel = float(temps[on_panel].mean()) if on_panel.sum() >= MIN_PANEL_PX else float("nan")
    return cell_t, count, panel, int(on_panel.sum()), cell_idx, on_panel


# ---------------------------------------------------------------------------------------------- review

def review_image(frame_bgr: np.ndarray, H: np.ndarray, uv: np.ndarray, temps: np.ndarray, on_panel, cell_idx,
                 cell_t, panel, spec: PanelSpec, lo: float, hi: float) -> np.ndarray:
    """Frame with each panel thermal pixel's footprint painted in its temperature, grid, and cell readings."""
    from segkit.panel.label import draw_cells
    out = frame_bgr.copy()
    paint = out.copy()
    cmap = cv2.applyColorMap(((np.clip((temps - lo) / max(hi - lo, 1e-6), 0, 1)) * 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)
    corners = np.array([0, 2, 8, 6])        # footprint rays at the 4 outer corners, in order
    for y, x in zip(*np.nonzero(on_panel)):
        q = G.project(H, uv[y, x, corners])
        if np.isfinite(q).all():
            cv2.fillConvexPoly(paint, np.round(q).astype(np.int32), tuple(int(c) for c in cmap[y, x]))
    out = cv2.addWeighted(paint, 0.65, out, 0.35, 0)
    draw_cells(out, H, spec, colour=(255, 255, 255))
    h, w = out.shape[:2]
    for r in range(spec.rows):
        for c in range(spec.cols):
            if np.isnan(cell_t[r, c]):
                continue
            x, y = G.project(H, np.array([[c + 0.5, r + 0.85]]))[0]
            if 0 <= x < w and 0 <= y < h:
                hot = abs(cell_t[r, c] - panel) > HOTSPOT_DELTA_C
                size = float(np.clip(G.cell_px(H, (c, r)) / 150, 0.45, 1.6))
                cv2.putText(out, f"{cell_t[r, c]:.1f}C", (int(x - 45 * size), int(y)), 0, size,
                            (0, 0, 255) if hot else (255, 255, 0), max(1, int(2 * size)))
    cv2.putText(out, f"panel avg {panel:.1f} C", (10, h - 20), 0, 1.0, (255, 255, 255), 2)
    return out


def summary_image(summary: dict, spec: PanelSpec, path: Path) -> None:
    """The panel as a 4 x 9 grid: each cell's temperature relative to the panel average, hotspots outlined."""
    s = 90
    img = np.full((spec.rows * s + 60, spec.cols * s * 2, 3), 255, np.uint8)
    for cell in summary["cells"]:
        r, c = cell["row"], cell["col"]
        x0, y0 = c * s * 2, 50 + r * s
        if cell["delta_c"] is None:
            colour = (200, 200, 200)
        else:
            d = np.clip(cell["delta_c"] / (2 * HOTSPOT_DELTA_C), -1, 1)
            colour = (int(255 * (1 - max(d, 0))), int(255 * (1 - abs(d))), int(255 * (1 + min(d, 0))))
        cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + 2 * s - 2, y0 + s - 2), colour, -1)
        if cell["hotspot"]:
            cv2.rectangle(img, (x0 + 2, y0 + 2), (x0 + 2 * s - 2, y0 + s - 2), (0, 0, 255), 4)
        txt = "unseen" if cell["temp_c"] is None else f"{cell['temp_c']:.1f}C"
        cv2.putText(img, f"{r},{c}", (x0 + 8, y0 + 26), 0, 0.6, (0, 0, 0), 1)
        cv2.putText(img, txt, (x0 + 8, y0 + 60), 0, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"panel avg {summary['panel_avg_c']:.1f} C, hotspot = |cell - avg| > {HOTSPOT_DELTA_C:g} C",
                (8, 32), 0, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-panel-thermal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--calib", type=Path, required=True, help="thermal_calib.json (segkit.thermal_calib)")
    ap.add_argument("--source", choices=["labels", "tracker"], default="labels",
                    help="panel geometry from panel_labels.npz or the tracker's track.npz")
    ap.add_argument("--latency-ms", type=float, help="override the calibration's thermal latency")
    ap.add_argument("--review-every", type=int, default=60)
    args = ap.parse_args()

    frames, Hs, valid, spec = load_session(args.session)
    if args.source == "tracker":
        tr = np.load(args.session / "track.npz")
        Hs = tr["H"]
        valid = np.isfinite(Hs).all(axis=(1, 2))
    calib = load_calib(args.calib)
    latency = (args.latency_ms if args.latency_ms is not None else calib["latency_ms"]) * 1e6
    K, _, _ = video_geometry(args.session)
    vt = video_times(args.session, len(frames))
    t_arr, temps = thermal_stream(args.session)
    rays = footprint_rays(calib)
    lo, hi = np.percentile(temps, [2, 98])

    out_csv = args.session / f"thermal_cells{'_tracker' if args.source == 'tracker' else ''}.csv"
    review = args.session / f"thermal_review{'_tracker' if args.source == 'tracker' else ''}"
    review.mkdir(exist_ok=True)
    cols = [f"r{r}c{c}" for r in range(spec.rows) for c in range(spec.cols)]
    per_frame = []
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "packet", "packet_dt_ms", "panel_avg_c", "panel_px", "n_cells", "hotspots"]
                   + cols + [f"n_{c}" for c in cols])
        for i in tqdm(np.flatnonzero(valid)):
            want = vt[i] + latency
            k = int(np.argmin(np.abs(t_arr - want)))
            dt = (t_arr[k] - want) / 1e6
            if abs(dt) > MAX_PACKET_DT_MS:
                continue
            t, r1, r2 = panel_pose(Hs[i], K, spec)
            uv = cast(rays, calib, t, r1, r2, spec)
            cell_t, count, panel, n_px, cell_idx, on_panel = frame_readings(uv, temps[k], spec)
            if not np.isfinite(panel):
                continue
            hot = [f"{r},{c}" for r, c in zip(*np.nonzero(np.abs(cell_t - panel) > HOTSPOT_DELTA_C))]
            w.writerow([i + 1, f"{(vt[i] - vt[0]) / 1e9:.3f}", k, f"{dt:.0f}", f"{panel:.2f}", n_px,
                        int(np.isfinite(cell_t).sum()), " ".join(hot)]
                       + [("" if np.isnan(x) else f"{x:.2f}") for x in cell_t.ravel()] + list(count.ravel()))
            per_frame.append((i, panel, cell_t, count))
            if len(per_frame) % args.review_every == 1:
                img = review_image(cv2.imread(str(frames[i])), Hs[i], uv, temps[k], on_panel, cell_idx,
                                   cell_t, panel, spec, lo, hi)
                cv2.imwrite(str(review / f"{i + 1:05d}.jpg"), img)

    # Session summary: per cell, median over frames of (cell - that frame's panel average).
    panels = np.array([p for _, p, _, _ in per_frame])
    deltas = np.stack([c - p for _, p, c, _ in per_frame])
    seen = np.isfinite(deltas).sum(0)
    avg = float(np.median(panels))
    cells = []
    for r in range(spec.rows):
        for c in range(spec.cols):
            d = deltas[:, r, c]
            d = d[np.isfinite(d)]
            delta = float(np.median(d)) if len(d) else None
            cells.append({"row": r, "col": c, "frames": int(seen[r, c]),
                          "temp_c": None if delta is None else round(avg + delta, 2),
                          "delta_c": None if delta is None else round(delta, 2),
                          "hotspot": delta is not None and abs(delta) > HOTSPOT_DELTA_C,
                          "hotspot_frames": int((np.abs(d) > HOTSPOT_DELTA_C).sum())})
    summary = {"session": args.session.name, "source": args.source, "frames": len(per_frame),
               "panel_avg_c": round(avg, 2), "panel_avg_range_c": [round(float(x), 2) for x in
                                                                    np.percentile(panels, [5, 95])],
               "hotspot_delta_c": HOTSPOT_DELTA_C, "latency_ms": latency / 1e6,
               "hotspots": [f"{x['row']},{x['col']}" for x in cells if x["hotspot"]], "cells": cells}
    suffix = "_tracker" if args.source == "tracker" else ""
    (args.session / f"thermal_summary{suffix}.json").write_text(json.dumps(summary, indent=1))
    summary_image(summary, spec, args.session / f"thermal_summary{suffix}.jpg")
    unseen = sum(x["temp_c"] is None for x in cells)
    print(f"{len(per_frame)} frames with panel temperatures; panel average {avg:.1f} C "
          f"(5-95%: {summary['panel_avg_range_c']}); {len(cells) - unseen}/{len(cells)} cells seen; "
          f"hotspots (|cell - avg| > {HOTSPOT_DELTA_C:g} C): {summary['hotspots'] or 'none'}")


if __name__ == "__main__":
    main()
