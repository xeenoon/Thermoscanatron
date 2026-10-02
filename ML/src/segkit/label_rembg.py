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
    args = p.parse_args()

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


if __name__ == "__main__":
    main()
