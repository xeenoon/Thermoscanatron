"""Auto-label captured hand photos: MediaPipe decides if there is a hand, rembg draws its outline.

segkit-label data/captures --out data/hands_v1

For every *.jpg under the input dir (recursively) writes:
  images/<stem>.jpg     EXIF-rotated RGB photo (what the mask is aligned to)
  masks/<stem>.png      0/255 hand mask; all zeros for a no-hand frame
  overlays/<stem>.jpg   downscaled photo + mask outline + landmarks, for eyeballing
  index.csv             per-image stats and QC flags
Already-labelled stems are skipped, so it can be re-run as new captures arrive.

With the hand check (default), a frame where MediaPipe finds no hand becomes a negative sample
(empty mask, flag no_hand), and only rembg blobs that contain hand landmarks are kept, so a door
or chair that rembg also picked out is dropped. --no-hand-check keeps the largest rembg blob.

rembg outlines whole foreground objects, so a bare forearm, a face or a body touching the hand comes back as
part of the same blob. Each hand's mask is therefore limited to (see restrict_to_hands):
  - the side of the wrist towards the fingers: a straight cut across the wrist, perpendicular to the
    wrist -> middle-knuckle direction (the eval set's "hand" definition, ML/README.md);
  - a box around that hand's landmarks, grown by HAND_BOX_GROW of its size (drops faces and bodies).
Frames where that removed a lot (> CUT_FLAG of the blob) are flagged "cut" for review.
--recut re-applies this to an existing dataset's masks (re-detecting landmarks) without re-running rembg.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch  # noqa: F401  loads the CUDA/cuDNN libs that onnxruntime-gpu links against, when present
from PIL import Image, ImageOps
from rembg import new_session, remove

MIN_AREA, MAX_AREA = 0.01, 0.6     # mask fraction of the frame outside this range -> flagged
MIN_EDGE_SHARPNESS = 20.0          # Laplacian variance along the outline below this -> flagged blurry
MAX_DISCARDED = 0.02               # foreground dropped as non-hand blobs, above this -> flagged
MIN_LANDMARKS_INSIDE = 0.9         # fraction of hand landmarks inside the mask, below this -> flagged
LANDMARK_TOLERANCE_PX = 6          # fingertip landmarks sit on the edge: test against a dilated mask
OVERLAY_SIDE = 640
WRIST_BACK = 0.15                  # cut this far (in palm lengths) past the wrist landmark, towards the arm
HAND_BOX_GROW = 0.25               # landmark box grown by this fraction of its larger side, each way
CUT_FLAG = 0.15                    # share of the rembg blob removed by restrict_to_hands -> flagged "cut"
FIELDS = ["stem", "source", "width", "height", "hands", "area", "discarded", "lm_in_mask", "edge_sharpness",
          "flags", "model"]


def keep_components(alpha: np.ndarray, landmarks: list[np.ndarray] | None) -> tuple[np.ndarray, float]:
    """Threshold soft alpha and keep the hand blobs.

    With landmarks: keep every component touching a landmark. Without: keep the largest component.
    Returns (mask 0/255, discarded fraction of foreground).
    """
    binary = (alpha >= 128).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary), 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    if landmarks:
        h, w = cc.shape
        r = LANDMARK_TOLERANCE_PX
        ids = set()
        for x, y in np.concatenate(landmarks).round().astype(int):
            if 0 <= x < w and 0 <= y < h:
                win = cc[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
                ids |= set(np.unique(win[win > 0]).tolist())
        keep = sorted(ids)
    else:
        keep = [1 + int(np.argmax(areas))]
    if not keep:
        return np.zeros_like(binary), 1.0
    kept = sum(areas[k - 1] for k in keep)
    return np.isin(cc, keep).astype(np.uint8) * 255, float(1 - kept / areas.sum())


def restrict_to_hands(mask: np.ndarray, landmarks: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Keep only the part of the mask each hand can own: fingers' side of a wrist cut, inside a box around the
    hand. Returns (mask, share of the input mask removed). MediaPipe landmarks: 0 wrist, 9 middle-finger MCP."""
    if not landmarks or not mask.any():
        return mask, 0.0
    h, w = mask.shape
    ys, xs = np.mgrid[0:h, 0:w]
    allowed = np.zeros((h, w), bool)
    for lm in landmarks:
        wrist, mcp = lm[0], lm[9]
        d = mcp - wrist
        palm = float(np.hypot(*d))
        if palm < 1:
            continue
        d /= palm
        c = wrist - WRIST_BACK * palm * d
        side = (xs - c[0]) * d[0] + (ys - c[1]) * d[1] >= 0
        x0, y0 = lm.min(0)
        x1, y1 = lm.max(0)
        g = HAND_BOX_GROW * max(x1 - x0, y1 - y0)
        box = (xs >= x0 - g) & (xs <= x1 + g) & (ys >= y0 - g) & (ys <= y1 + g)
        allowed |= side & box
    before = int((mask > 0).sum())
    out = np.where(allowed, mask, 0).astype(np.uint8)
    return out, 1 - int((out > 0).sum()) / max(before, 1)


def landmarks_inside(mask: np.ndarray, landmarks: list[np.ndarray]) -> float:
    """Fraction of in-frame landmarks on the mask (MediaPipe extrapolates fingertips past the frame edge)."""
    k = 2 * LANDMARK_TOLERANCE_PX + 1
    grown = cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0
    h, w = mask.shape
    pts = np.concatenate(landmarks).round().astype(int)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    return float(grown[pts[:, 1], pts[:, 0]].mean()) if len(pts) else 0.0


def edge_sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance in a thin band around the outline: low = motion blur / soft edge."""
    k = np.ones((7, 7), np.uint8)
    band = cv2.dilate(mask, k) > cv2.erode(mask, k)
    if not band.any():
        return 0.0
    lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return float(lap[band].var())


def overlay(rgb: np.ndarray, mask: np.ndarray, landmarks: list[np.ndarray] | None) -> np.ndarray:
    scale = OVERLAY_SIDE / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    vis[m > 0] = (0.7 * vis[m > 0] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    for hand in landmarks or []:
        for x, y in hand * scale:
            cv2.circle(vis, (int(x), int(y)), 2, (255, 0, 255), -1)
    if landmarks is not None and not landmarks:
        cv2.putText(vis, "NO HAND", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return vis


def load_index(index_path: Path) -> list[dict]:
    """Existing rows, upgraded in place to the current columns (older indexes lack hands/lm_in_mask)."""
    if not index_path.exists():
        return []
    with index_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    if header != FIELDS:
        rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
        with index_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-label")
    p.add_argument("captures", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/hands_v1"))
    p.add_argument("--model", default="birefnet-general-lite",
                   help="rembg model, e.g. birefnet-general-lite, birefnet-general, isnet-general-use")
    p.add_argument("--no-hand-check", action="store_true", help="skip MediaPipe; keep the largest rembg blob")
    p.add_argument("--limit", type=int, default=0, help="only label this many new images (0 = all)")
    p.add_argument("--recut", action="store_true",
                   help="re-apply the wrist cut / hand box to the existing masks in --out (no rembg)")
    args = p.parse_args()
    if args.recut:
        return recut(args.out)

    for sub in ("images", "masks", "overlays"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.csv"
    done = {row["stem"] for row in load_index(index_path)}

    todo = [p for p in sorted(args.captures.rglob("*.jpg")) if p.stem not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already labelled, {len(todo)} to do, model {args.model}, "
          f"hand check {'off' if args.no_hand_check else 'on'}")
    if not todo:
        return

    detector = None
    if not args.no_hand_check:
        from segkit.hand_landmarks import HandLandmarks
        detector = HandLandmarks()
    # Prefer CUDA when onnxruntime-gpu is installed; skip TensorRT (not installed, slow to fail over).
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    print(f"onnxruntime providers: {providers}")
    session = new_session(args.model, providers=providers)

    new_file = not index_path.exists()
    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for i, src in enumerate(todo, 1):
            t0 = time.perf_counter()
            img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            rgb = np.asarray(img)
            landmarks = detector.detect(rgb) if detector else None

            flags = []
            lm_in = None
            if landmarks is not None and not landmarks:
                mask, discarded = np.zeros(rgb.shape[:2], np.uint8), 0.0
                flags.append("no_hand")
            else:
                alpha = np.asarray(remove(img, session=session, only_mask=True))
                mask, discarded = keep_components(alpha, landmarks)
                if landmarks:
                    mask, removed = restrict_to_hands(mask, landmarks)
                    if removed > CUT_FLAG:
                        flags.append("cut")
                area = float((mask > 0).mean())
                if not MIN_AREA <= area <= MAX_AREA:
                    flags.append("area")
                if discarded > MAX_DISCARDED:
                    flags.append("extra_blobs")
                if landmarks:
                    lm_in = landmarks_inside(mask, landmarks)
                    if lm_in < MIN_LANDMARKS_INSIDE:
                        flags.append("landmarks_outside")
            area = float((mask > 0).mean())
            sharp = edge_sharpness(rgb, mask)
            if "no_hand" not in flags and sharp < MIN_EDGE_SHARPNESS:
                flags.append("blurry")

            img.save(args.out / "images" / f"{src.stem}.jpg", quality=95)
            cv2.imwrite(str(args.out / "masks" / f"{src.stem}.png"), mask)
            cv2.imwrite(str(args.out / "overlays" / f"{src.stem}.jpg"), overlay(rgb, mask, landmarks))
            writer.writerow({"stem": src.stem, "source": str(src), "width": rgb.shape[1], "height": rgb.shape[0],
                             "hands": "" if landmarks is None else len(landmarks), "area": f"{area:.4f}",
                             "discarded": f"{discarded:.4f}", "lm_in_mask": "" if lm_in is None else f"{lm_in:.2f}",
                             "edge_sharpness": f"{sharp:.1f}", "flags": "|".join(flags), "model": args.model})
            f.flush()
            print(f"[{i}/{len(todo)}] {src.stem} {rgb.shape[1]}x{rgb.shape[0]} hands "
                  f"{'-' if landmarks is None else len(landmarks)} area {area:.3f} "
                  f"{' '.join(flags) or 'ok'} ({time.perf_counter() - t0:.1f}s)", flush=True)
    if detector:
        detector.close()


def recut(dataset: Path) -> None:
    """Wrist-cut / hand-box the masks of an already-labelled dataset in place; old masks kept in masks_uncut/."""
    from segkit.hand_landmarks import HandLandmarks
    detector = HandLandmarks()
    (dataset / "masks_uncut").mkdir(exist_ok=True)
    rows = load_index(dataset / "index.csv")
    changed = 0
    for i, row in enumerate(rows, 1):
        mpath = dataset / "masks" / f"{row['stem']}.png"
        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None or not mask.any():
            continue
        rgb = np.asarray(ImageOps.exif_transpose(Image.open(dataset / "images" / f"{row['stem']}.jpg")).convert("RGB"))
        landmarks = detector.detect(rgb)
        new, removed = restrict_to_hands(mask, landmarks)
        flags = [f for f in row["flags"].split("|") if f and f != "cut"]
        if removed > 0:
            backup = dataset / "masks_uncut" / mpath.name
            if not backup.exists():
                cv2.imwrite(str(backup), mask)
            cv2.imwrite(str(mpath), new)
            cv2.imwrite(str(dataset / "overlays" / f"{row['stem']}.jpg"), overlay(rgb, new, landmarks))
            changed += 1
        if removed > CUT_FLAG:
            flags.append("cut")
        row["flags"] = "|".join(flags)
        if i % 200 == 0:
            print(f"[{i}/{len(rows)}] {changed} masks cut", flush=True)
    with (dataset / "index.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    detector.close()
    print(f"{changed} of {len(rows)} masks cut at the wrist / hand box")


if __name__ == "__main__":
    main()
