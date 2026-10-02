"""Auto-label captured hand photos with rembg (background removal).

segkit-label data/captures --out data/hands_v1

For every *.jpg under the input dir (recursively) writes:
  images/<stem>.jpg     EXIF-rotated RGB photo (what the mask is aligned to)
  masks/<stem>.png      0/255 hand mask (largest foreground component)
  overlays/<stem>.jpg   downscaled photo + mask outline, for eyeballing
  index.csv             per-image stats and QC flags
Already-labelled stems are skipped, so it can be re-run as new captures arrive.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from rembg import new_session, remove

MIN_AREA, MAX_AREA = 0.01, 0.6     # mask fraction of the frame outside this range -> flagged
MIN_EDGE_SHARPNESS = 20.0          # Laplacian variance along the outline below this -> flagged blurry
OVERLAY_SIDE = 640


def clean_mask(alpha: np.ndarray) -> tuple[np.ndarray, int]:
    """Threshold soft alpha, keep the largest connected component. Returns (mask, n_components)."""
    binary = (alpha >= 128).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary), 0
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (cc == largest).astype(np.uint8) * 255, n - 1


def edge_sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Laplacian variance in a thin band around the outline: low = motion blur / soft edge."""
    k = np.ones((7, 7), np.uint8)
    band = cv2.dilate(mask, k) > cv2.erode(mask, k)
    if not band.any():
        return 0.0
    lap = cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)
    return float(lap[band].var())


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    scale = OVERLAY_SIDE / max(rgb.shape[:2])
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
    vis[m > 0] = (0.7 * vis[m > 0] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
    return vis


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-label")
    p.add_argument("captures", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/hands_v1"))
    p.add_argument("--model", default="birefnet-general-lite",
                   help="rembg model, e.g. birefnet-general-lite, birefnet-general, isnet-general-use")
    p.add_argument("--limit", type=int, default=0, help="only label this many new images (0 = all)")
    args = p.parse_args()

    for sub in ("images", "masks", "overlays"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.csv"
    fields = ["stem", "source", "width", "height", "area", "components", "edge_sharpness", "flags", "model"]
    done = set()
    if index_path.exists():
        with index_path.open() as f:
            done = {row["stem"] for row in csv.DictReader(f)}

    todo = [p for p in sorted(args.captures.rglob("*.jpg")) if p.stem not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already labelled, {len(todo)} to do, model {args.model}")
    if not todo:
        return

    session = new_session(args.model)
    new_file = not index_path.exists()
    with index_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for i, src in enumerate(todo, 1):
            t0 = time.perf_counter()
            img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
            rgb = np.asarray(img)
            alpha = np.asarray(remove(img, session=session, only_mask=True))
            mask, n_comp = clean_mask(alpha)
            area = float((mask > 0).mean())
            sharp = edge_sharpness(rgb, mask)

            flags = []
            if not MIN_AREA <= area <= MAX_AREA:
                flags.append("area")
            if n_comp > 1:
                flags.append("multi_component")
            if sharp < MIN_EDGE_SHARPNESS:
                flags.append("blurry")

            img.save(args.out / "images" / f"{src.stem}.jpg", quality=95)
            cv2.imwrite(str(args.out / "masks" / f"{src.stem}.png"), mask)
            cv2.imwrite(str(args.out / "overlays" / f"{src.stem}.jpg"), overlay(rgb, mask))
            writer.writerow({"stem": src.stem, "source": str(src), "width": rgb.shape[1], "height": rgb.shape[0],
                             "area": f"{area:.4f}", "components": n_comp, "edge_sharpness": f"{sharp:.1f}",
                             "flags": "|".join(flags), "model": args.model})
            f.flush()
            print(f"[{i}/{len(todo)}] {src.stem} {rgb.shape[1]}x{rgb.shape[0]} area {area:.3f} "
                  f"sharp {sharp:.0f} {' '.join(flags) or 'ok'} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
