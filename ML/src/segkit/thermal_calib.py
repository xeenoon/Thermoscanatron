"""Fit the phone-camera <-> thermal-camera alignment from a recorded session (hand silhouettes).

Registration idea after Hong et al. (2022): the RGB image carries the geometry (here: the hand mask from
HandSegNet), the thermal image carries the temperatures, and a calibrated mapping between the two lets
each be read in the other's coordinates.
  F. Hong, J. Song, H. Meng, R. Wang, F. Fang, G. Zhang, "A novel framework on intelligent detection for
  module defects of PV plant combining the visible and infrared images", Solar Energy 236 (2022) 406-416,
  doi:10.1016/j.solener.2022.03.018.
Their registration is a homography from matched image features; a 32x24 thermal image has no usable
features and a phone-mounted sensor is not a planar scene, so this fits a full two-camera pose instead.

Model
  phone camera  pinhole; intrinsics from Camera2 (meta.json); coordinates of the upright analysis frame
                (x right, y down, z forward), centimetres.
  thermal       MLX90640ESF-BAB (Adafruit 4407), equidistant: angle from axis = radius / f, focal lengths
                from the MEASURED field of view (FOV_X_DEG x FOV_Y_DEG, see below) times one fitted scale k;
                mirrored readout (datasheet Figure 3: column 1 is on the right as the sensor looks out).
  pose          thermal axes in camera coordinates R = Ry(yaw) @ Rx(pitch) @ Rz(roll): yaw + turns the
                thermal camera right, pitch + up, roll + clockwise as seen from behind the phone.
                Thermal centre at c = (x, y, z) cm: x right, y down, z forward of the phone lens.
  hand          per frame, a plane at depth Z estimated from the mask area (open hand ~HAND_AREA_CM2).

Nothing about the mounting is assumed: any roll, large yaw/pitch, tens of cm of offset, mirrored or not.
  0. Latency (thermal arrival after camera capture) from geometry-free signals: hand size in the camera and
     warm area in the thermal image rise and fall together as the hand moves nearer/further, whatever the
     mounting; the cross-correlation peak (100-350 ms) is the lag, else DEFAULT_LATENCY_MS. Fitting it with
     the pose instead lets a lag on a circling hand pose as a roll (a delayed circle is a rotated circle).
  1. Global pose from points: per frame, the camera hand centroid at depth Z is a 3D point and the warm-blob
     centroid is its thermal image. Pose from those 3D-2D pairs is solved by robust Levenberg-Marquardt
     from a grid of starting rotations (all rolls, yaw/pitch +-60 deg), for both mirror hypotheses.
     Hands cut off by the crop box are skipped (their area, hence depth, is wrong).
  2. Silhouette refinement: for every thermal pixel (2x2 sub-rays, blurred by the optics' PSF) the ray is
     intersected with the hand plane and looked up in the camera's hand mask, predicting its "hand
     fraction"; pose + focal scale are tuned to maximise the correlation with the pixel warmth.
     Correlation ignores warmth scale/offset, so the warm forearm or a blur halo only lowers the score.
Translation is separated from rotation by parallax, i.e. by the hand being at different depths: move the
hand nearer and further while recording.

Physical constraints (not mounting assumptions) rule out impossible poses, in particular the planar twin:
with the hand at one depth the points are near-coplanar, and a camera reflected through that plane,
looking back at it, explains them equally well (mirrored, ~twice the hand depth in front, facing the
phone). Both cameras must look the same way (optical axes within 80 deg), the thermal camera must be on the
phone's side of the hand and within MAX_OFFSET_CM of the lens, and its lens within +-33% of the datasheet.
A tilt and a sideways offset look alike when the hand stays at one depth, so both stages also carry a weak
preference for small offsets and a datasheet lens; depth variation (parallax) outweighs it.

The same algorithm runs on the phone in android-app ThermalCalibration.kt.

    uv run python -m segkit.thermal_calib data/thermal_sessions/<session> --t-ranges 0-55,60-70
"""

import argparse
import csv
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, label
from scipy.optimize import least_squares, minimize

TW, TH = 32, 24
T_CX, T_CY = (TW - 1) / 2, (TH - 1) / 2
# Field of view MEASURED on this sensor (soldering-iron point source, session 20261002_171833, 126 pairs,
# equidistant model, bootstrap 95%): 49.7 (49.3-50.0) x 38.3 (37.9-39.0) deg. The datasheet's "typical"
# 55 x 35 deg (Table 15) is defined at the 50%-sensitivity edge of the whole array, not by pixel-centre
# spacing, and put the hand calibration ~2-3 thermal px off. Lens shape: equidistant fits slightly better
# than pinhole (median 0.44 vs 0.47 px); the datasheet does not specify it.
FOV_X_DEG, FOV_Y_DEG = 49.7, 38.3
T_FX = (TW / 2) / np.radians(FOV_X_DEG / 2)    # px per radian, horizontal (32 px)
T_FY = (TH / 2) / np.radians(FOV_Y_DEG / 2)    # px per radian, vertical (24 px)
BLUR_PX = 0.8                           # thermal optics point-spread, in thermal pixels
HAND_AREA_CM2 = 130.0                   # silhouette area of an open adult hand, palm + fingers
MASK_GRID = 96                          # crop mask resolution used for fitting (384 / 4)
RECORD_BYTES = 8 + 1566                 # thermal.bin record: int64 arrival ns + THM2 packet
MIN_CONTRAST_C = 4.0                    # thermal frames flatter than this hold no hand
MAX_EDGE_FRACTION = 0.15                # hand cut off by the crop box: area (and depth) unreliable
DEFAULT_LATENCY_MS = 225                # measured on the first recorded session (firmware + USB property)
SUB = np.array([[-0.25, -0.25], [0.25, -0.25], [-0.25, 0.25], [0.25, 0.25]])
MAX_OFFSET_CM = 40.0                    # thermal camera within this distance of the phone lens
MIN_AXIS_COS = np.cos(np.radians(80))   # both cameras face the same way
MAX_LOG_K = np.log(1.33)                # lens within +-33% of the datasheet field of view
OFFSET_PRIOR_CM = 15.0                  # weak tie-breaker towards small offsets (tilt vs shift at one depth)
LENS_PRIOR = 0.2
PRIOR_WEIGHT = 0.01                     # in correlation units: a 20 cm offset costs ~0.02


def prior(p: np.ndarray) -> float:
    return PRIOR_WEIGHT * (np.sum(p[3:6] ** 2) / OFFSET_PRIOR_CM ** 2 + p[6] ** 2 / LENS_PRIOR ** 2)


def plausible(p: np.ndarray, near_cm: float) -> bool:
    """Physically possible rig: same viewing direction, sensor near the phone and behind the hand."""
    return bool(rotation(*p[:3])[2, 2] > MIN_AXIS_COS and np.linalg.norm(p[3:6]) < MAX_OFFSET_CM
                and p[5] < near_cm - 5 and abs(p[6]) < MAX_LOG_K)


# ---------------------------------------------------------------------------------------------- data

@dataclass
class Session:
    f: float
    cx: float
    cy: float
    t_ns: np.ndarray           # thermal arrival times
    temps: np.ndarray          # (n, 24, 32) degC
    cam_ts: np.ndarray         # camera sensor timestamps
    rows: list[dict]
    dir: Path


def load(session: Path) -> Session:
    meta = json.loads((session / "meta.json").read_text())
    cam, an = meta["camera"], meta["analysis"]
    # Focal length in sensor pixels, then into analysis-buffer pixels (rotation keeps it: square pixels).
    f = cam["focal_lengths_mm"][0] / cam["sensor_physical_mm"][0] * cam["pixel_array"][0]
    f *= an["sensor_to_buffer"][0]
    raw = (session / "thermal.bin").read_bytes()
    n = len(raw) // RECORD_BYTES
    t_ns = np.array([struct.unpack_from("<q", raw, i * RECORD_BYTES)[0] for i in range(n)])
    temps = np.stack([np.frombuffer(raw, "<i2", TW * TH, i * RECORD_BYTES + 8 + 28).reshape(TH, TW) / 100
                      for i in range(n)])
    rows = list(csv.DictReader((session / "frames.csv").open()))
    cam_ts = np.array([int(r["sensor_ts_ns"]) for r in rows])
    return Session(f, an["upright_w"] / 2, an["upright_h"] / 2, t_ns, temps, cam_ts, rows, session)


def warmth(t: np.ndarray) -> np.ndarray | None:
    """Thermal frame -> 0 (background) .. 1 (fully hand) per pixel, or None if nothing is warm."""
    lo, hi = np.percentile(t, 20), np.percentile(t, 98)
    if hi - lo < MIN_CONTRAST_C:
        return None
    return np.clip((t - lo) / (hi - lo), 0, 1)


def blob_centroid(img: np.ndarray, threshold: float) -> tuple[float, float] | None:
    """Weighted centroid (x, y) of the largest 8-connected region above threshold."""
    lab, n = label(img > threshold, structure=np.ones((3, 3)))
    if n == 0:
        return None
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    w = np.where(lab == big, img, 0)
    y, x = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float((w * x).sum() / w.sum()), float((w * y).sum() / w.sum())


@dataclass
class Pairs:
    masks: np.ndarray      # (n, 96, 96) hand probability in the crop
    box: np.ndarray        # (n, 3) left, top, side in upright frame px
    depth: np.ndarray      # (n,) cm
    warm: np.ndarray       # (n, 768) observed warmth
    cam_pt: np.ndarray     # (n, 3) hand centroid in camera coordinates, cm
    th_pt: np.ndarray      # (n, 2) warm-blob centroid, thermal pixels (u, v)
    cam_idx: np.ndarray
    th_idx: np.ndarray

    def every(self, k: int) -> "Pairs":
        return Pairs(*(a[::max(1, k)] for a in vars(self).values()))


def make_pairs(s: Session, dt_ms: float, t_ranges: list[tuple[float, float]] | None) -> Pairs:
    """One pair per thermal frame: the camera frame captured dt_ms before the packet arrived."""
    t0 = s.cam_ts[0]
    out = {k: [] for k in Pairs.__dataclass_fields__}
    for k, arrive in enumerate(s.t_ns):
        target = arrive - dt_ms * 1e6
        i = int(np.clip(np.searchsorted(s.cam_ts, target), 1, len(s.cam_ts) - 1))
        i = i if abs(s.cam_ts[i] - target) < abs(s.cam_ts[i - 1] - target) else i - 1
        if abs(s.cam_ts[i] - target) > 40e6:
            continue
        if t_ranges and not any(a <= (s.cam_ts[i] - t0) / 1e9 < b for a, b in t_ranges):
            continue
        r = s.rows[i]
        if r["hand_visible"] != "1":
            continue
        w = warmth(s.temps[k])
        if w is None:
            continue
        left, top, side = int(r["box_left"]), int(r["box_top"]), int(r["box_side"])
        area_px = float(r["mask_frac"]) * side * side
        if area_px < 500:
            continue
        m = cv2.imread(str(s.dir / "masks" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
        m = cv2.resize(m, (MASK_GRID, MASK_GRID), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]]) > 0.5
        if border.sum() > MAX_EDGE_FRACTION * 4 * MASK_GRID:
            continue
        mc, tc = blob_centroid(m, 0.5), blob_centroid(w, 0.5)
        if mc is None or tc is None:
            continue
        z = s.f * np.sqrt(HAND_AREA_CM2 / area_px)
        px = left + (mc[0] + 0.5) / MASK_GRID * side
        py = top + (mc[1] + 0.5) / MASK_GRID * side
        out["masks"].append(m)
        out["box"].append((left, top, side))
        out["depth"].append(z)
        out["warm"].append(w.ravel())
        out["cam_pt"].append(((px - s.cx) / s.f * z, (py - s.cy) / s.f * z, z))
        out["th_pt"].append(tc)
        out["cam_idx"].append(i)
        out["th_idx"].append(k)
    return Pairs(**{k: np.array(v) for k, v in out.items()})


# ---------------------------------------------------------------------------------------------- geometry

def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def canonical(p: np.ndarray) -> np.ndarray:
    """Same rotation, reported with |pitch| <= 90 deg: (yaw, pitch, roll) == (yaw+180, 180-pitch, roll+180)."""
    q = np.array(p, float)
    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    q[1] = wrap(q[1])
    if abs(q[1]) > np.pi / 2:
        q[0], q[1], q[2] = q[0] + np.pi, np.pi - q[1], q[2] + np.pi
    q[:3] = [wrap(a) for a in q[:3]]
    return q


def project_thermal(pts: np.ndarray, p: np.ndarray, mirror: bool) -> np.ndarray:
    """Camera-frame points (n, 3) -> thermal pixels (n, 2); p = (yaw, pitch, roll, x, y, z, log k)."""
    q = (pts - p[3:6]) @ rotation(*p[:3])            # = R^T (P - c), row-wise
    r = np.hypot(q[:, 0], q[:, 1])
    th = np.arctan2(r, q[:, 2])
    s = np.where(r > 1e-9, th / np.maximum(r, 1e-9), 0)
    k = np.exp(p[6])
    u = T_CX + (-1 if mirror else 1) * T_FX * k * s * q[:, 0]
    v = T_CY + T_FY * k * s * q[:, 1]
    return np.stack([u, v], 1)


def thermal_rays(mirror: bool, k: float = 1.0) -> np.ndarray:
    """(768*4, 3) unit rays in thermal coordinates, 4 sub-rays per pixel, pixel-major."""
    v, u = np.mgrid[0:TH, 0:TW]
    uv = np.stack([u.ravel(), v.ravel()], 1)[:, None, :] + SUB[None]
    mx = (uv[..., 0] - T_CX) / (T_FX * k) * (-1 if mirror else 1)
    my = (uv[..., 1] - T_CY) / (T_FY * k)
    th = np.hypot(mx, my)
    s = np.where(th > 1e-9, np.sin(th) / np.maximum(th, 1e-9), 1.0)
    return np.stack([mx * s, my * s, np.cos(th)], -1).reshape(-1, 3)


# ---------------------------------------------------------------------------------------------- stage 1

def pose_from_points(pairs: Pairs) -> tuple[np.ndarray, bool, float]:
    """Robust 3D-2D pose (no mounting assumptions): multi-start LM over rolls x yaw/pitch, both mirrors.
    Returns params, mirror, median reprojection error (thermal px)."""
    def resid(p6, mirror):
        p = np.append(p6, 0.0)
        return np.concatenate([(project_thermal(pairs.cam_pt, p, mirror) - pairs.th_pt).ravel(),
                               p6[3:6] / OFFSET_PRIOR_CM])

    near = float(np.percentile(pairs.depth, 10))
    best = (np.inf, None, None)
    for mirror in (False, True):
        for roll in np.radians(np.arange(0, 360, 30)):
            for yaw in np.radians([-60, -30, 0, 30, 60]):
                for pitch in np.radians([-60, -30, 0, 30, 60]):
                    x0 = np.array([yaw, pitch, roll, 0, 0, 0.0])
                    r = least_squares(resid, x0, args=(mirror,), loss="huber", f_scale=1.5, max_nfev=60)
                    if r.cost < best[0] and plausible(np.append(r.x, 0.0), near):
                        best = (r.cost, r.x, mirror)
    _, x, mirror = best
    err = np.median(np.linalg.norm(resid(x, mirror)[:-3].reshape(-1, 2), axis=1))
    x[2] = (x[2] + np.pi) % (2 * np.pi) - np.pi
    return np.append(x, 0.0), mirror, float(err)


# ---------------------------------------------------------------------------------------------- stage 2

def predict(p: np.ndarray, mirror: bool, s: Session, pairs: Pairs) -> tuple[np.ndarray, np.ndarray]:
    """Predicted hand fraction (n, 768) and validity (ray lands inside the crop box)."""
    yaw, pitch, roll, cx, cy, cz, logk = p
    d = thermal_rays(mirror, float(np.exp(logk))) @ rotation(yaw, pitch, roll).T
    n = len(pairs.depth)
    z = pairs.depth[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = (z - cz) / d[None, :, 2]                         # distance along each ray to the plane
        px = s.f * (cx + scale * d[None, :, 0]) / z + s.cx
        py = s.f * (cy + scale * d[None, :, 1]) / z + s.cy
    box = pairs.box.astype(float)
    gx = (px - box[:, :1]) / box[:, 2:] * MASK_GRID - 0.5
    gy = (py - box[:, 1:2]) / box[:, 2:] * MASK_GRID - 0.5
    inside = ((scale > 0) & (gx >= 0) & (gx <= MASK_GRID - 1) & (gy >= 0) & (gy <= MASK_GRID - 1))
    gx = np.clip(np.nan_to_num(gx), 0, MASK_GRID - 1.001)
    gy = np.clip(np.nan_to_num(gy), 0, MASK_GRID - 1.001)
    x0, y0 = gx.astype(int), gy.astype(int)
    fx, fy = gx - x0, gy - y0
    flat = pairs.masks.reshape(n, -1)
    def at(yy, xx):
        return np.take_along_axis(flat, yy * MASK_GRID + xx, 1)
    val = (at(y0, x0) * (1 - fx) * (1 - fy) + at(y0, x0 + 1) * fx * (1 - fy)
           + at(y0 + 1, x0) * (1 - fx) * fy + at(y0 + 1, x0 + 1) * fx * fy)
    frac = gaussian_filter(val.reshape(n, TH, TW, 4).mean(3), (0, BLUR_PX, BLUR_PX)).reshape(n, -1)
    valid = inside.reshape(n, TW * TH, 4).all(2)
    return frac, valid


def correlation(p, mirror, s, pairs) -> float:
    frac, valid = predict(p, mirror, s, pairs)
    if valid.sum() < 50:
        return -1.0
    c = np.corrcoef(frac[valid], pairs.warm[valid])[0, 1]
    return float(c) if np.isfinite(c) else -1.0


def refine(p0: np.ndarray, mirror: bool, s: Session, pairs: Pairs) -> np.ndarray:
    step = np.array([np.radians(3)] * 3 + [2.0] * 3 + [0.1])
    near = float(np.percentile(pairs.depth, 10))
    res = minimize(lambda q: -correlation(q, mirror, s, pairs) + prior(q) if plausible(q, near) else 1.0, p0,
                   method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([p0, p0 + np.diag(step)]),
                            "xatol": 1e-4, "fatol": 1e-5, "maxiter": 1500, "adaptive": True})
    return res.x


def sensitivity(p, mirror, s, pairs) -> np.ndarray:
    """Rough 1-sigma per parameter: the step that lowers the correlation by 0.002 (curvature of the peak)."""
    steps = np.array([np.radians(0.5)] * 3 + [0.5] * 3 + [0.02])
    c0 = correlation(p, mirror, s, pairs)
    out = []
    for i, h in enumerate(steps):
        e = np.zeros_like(p)
        e[i] = h
        curv = (2 * c0 - correlation(p + e, mirror, s, pairs) - correlation(p - e, mirror, s, pairs)) / h ** 2
        out.append(np.sqrt(0.004 / curv) if curv > 0 else np.inf)
    return np.array(out)


# ---------------------------------------------------------------------------------------------- stage 0

def estimate_latency(s: Session, t_ranges) -> int | None:
    """Hand size (camera) vs warm area (thermal) cross-correlation peak in 100-350 ms, if >= 0.6."""
    t0 = s.cam_ts[0]
    keep = [i for i, r in enumerate(s.rows) if r["hand_visible"] == "1"
            and (not t_ranges or any(a <= (s.cam_ts[i] - t0) / 1e9 < b for a, b in t_ranges))]
    if len(keep) < 10:
        return None
    ts = s.cam_ts[keep]
    area = np.array([float(s.rows[i]["mask_frac"]) * int(s.rows[i]["box_side"]) ** 2 for i in keep])
    warm = np.array([(t - np.percentile(t, 20) > MIN_CONTRAST_C).sum() for t in s.temps], float)
    grid = np.arange(ts[0], ts[-1], 20e6)
    a = np.interp(grid, ts, area)
    a -= a.mean()
    best, lag = -1.0, None
    for dt in range(100, 351, 10):
        b = np.interp(grid + dt * 1e6, s.t_ns, warm)
        b -= b.mean()
        c = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12)
        if c > best:
            best, lag = c, dt
    print(f"latency from hand size: {lag} ms (correlation {best:.2f})")
    return lag if best >= 0.6 else None


# ---------------------------------------------------------------------------------------------- driver

def fit(session: Path, t_ranges, out_dir: Path) -> dict:
    s = load(session)

    dt_ms = estimate_latency(s, t_ranges) or DEFAULT_LATENCY_MS
    pairs = make_pairs(s, float(dt_ms), t_ranges)
    p1, mirror, err1 = pose_from_points(pairs.every(len(pairs.depth) // 120))
    print(f"stage 1: latency {dt_ms} ms  mirror={mirror}  yaw/pitch/roll "
          f"{np.degrees(p1[:3]).round(1)} deg  xyz {p1[3:6].round(1)} cm  ({err1:.2f} px)")
    print(f"hand depth median {np.median(pairs.depth):.0f} cm (p10 {np.percentile(pairs.depth, 10):.0f}, "
          f"p90 {np.percentile(pairs.depth, 90):.0f})")

    fit_set = pairs.every(len(pairs.depth) // 200)
    c1 = correlation(p1, mirror, s, fit_set)
    p = canonical(refine(p1, mirror, s, fit_set))
    err = sensitivity(p, mirror, s, fit_set)
    corr = correlation(p, mirror, s, pairs)
    k = float(np.exp(p[6]))
    result = {
        "mirror": bool(mirror),
        "thermal_fx": T_FX * k, "thermal_fy": T_FY * k, "thermal_cx": T_CX, "thermal_cy": T_CY,
        "focal_scale": k,
        "yaw_deg": float(np.degrees(p[0])), "pitch_deg": float(np.degrees(p[1])), "roll_deg": float(np.degrees(p[2])),
        "x_cm": float(p[3]), "y_cm": float(p[4]), "z_cm": float(p[5]),
        "sigma": {"yaw_deg": float(np.degrees(err[0])), "pitch_deg": float(np.degrees(err[1])),
                  "roll_deg": float(np.degrees(err[2])), "x_cm": float(err[3]), "y_cm": float(err[4]),
                  "z_cm": float(err[5]), "focal_scale": float(k * err[6])},
        "latency_ms": dt_ms,
        "hand_area_cm2": HAND_AREA_CM2,
        "stage1_reprojection_px": err1,
        "stage1_correlation": c1,
        "correlation": corr,
        "pairs": int(len(pairs.depth)),
        "source": session.name,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thermal_calib.json").write_text(json.dumps(result, indent=2))
    sheet(p, mirror, s, pairs, out_dir / "check.jpg")
    return result


def sheet(p, mirror, s, pairs, path: Path, n: int = 12) -> None:
    """Per pair: camera crop on top; below, the thermal frame with the predicted hand outline (green)."""
    frac, _ = predict(p, mirror, s, pairs)
    tiles = []
    for k in np.linspace(0, len(pairs.depth) - 1, n).astype(int):
        crop = cv2.imread(str(s.dir / "crops" / f"{pairs.cam_idx[k]:06d}.jpg"))
        t = s.temps[pairs.th_idx[k]]
        tn = (np.clip((t - t.min()) / (np.ptp(t) + 1e-6), 0, 1) * 255).astype(np.uint8)
        th = cv2.applyColorMap(cv2.resize(tn, (256, 192), interpolation=cv2.INTER_CUBIC), cv2.COLORMAP_INFERNO)
        pm = cv2.resize((frac[k].reshape(TH, TW) * 255).astype(np.uint8), (256, 192), interpolation=cv2.INTER_CUBIC)
        cnts, _ = cv2.findContours((pm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(th, cnts, -1, (0, 255, 0), 2)
        tiles.append(np.vstack([cv2.resize(crop, (256, 256)), th]))
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--t-ranges", default="", help="seconds from session start to use, e.g. 0-55,60-70")
    ap.add_argument("--out", type=Path, default=Path("runs/thermal_calib"))
    a = ap.parse_args()
    ranges = [tuple(float(x) for x in r.split("-")) for r in a.t_ranges.split(",") if r] or None
    r = fit(a.session, ranges, a.out)
    sg = r["sigma"]
    print(f"yaw {r['yaw_deg']:.1f}±{sg['yaw_deg']:.1f}  pitch {r['pitch_deg']:.1f}±{sg['pitch_deg']:.1f}  "
          f"roll {r['roll_deg']:.1f}±{sg['roll_deg']:.1f} deg   mirror={r['mirror']}")
    print(f"x {r['x_cm']:.1f}±{sg['x_cm']:.1f}  y {r['y_cm']:.1f}±{sg['y_cm']:.1f}  "
          f"z {r['z_cm']:.1f}±{sg['z_cm']:.1f} cm   focal scale {r['focal_scale']:.3f}   "
          f"corr {r['stage1_correlation']:.3f} -> {r['correlation']:.3f}  ({r['pairs']} pairs, "
          f"latency {r['latency_ms']} ms)")


if __name__ == "__main__":
    main()
