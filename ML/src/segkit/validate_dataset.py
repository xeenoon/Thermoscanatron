"""Validate a labelled dataset dir (output of segkit-label) and render review contact sheets.

segkit-validate data/hands_v1

Checks: every index row has an image and a mask, same size, mask is binary 0/255 and non-empty,
no orphan files. Prints flag counts and writes <dataset>/review/sheet_NN.jpg (numbered tiles,
label colour: green ok, yellow flagged, red excluded) for eyeballing.
<dataset>/exclude.txt lists stems rejected in manual review (one per line, "# reason" optional).

For video frames (<video>_f<frame>) it also scores temporal consistency: a frame whose mask disagrees
with both neighbours while the neighbours agree with each other is a likely label glitch. The worst
are listed and drawn to review/outliers.jpg.
"""

import argparse
import collections
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np

TILE_W, TILE_H, COLS, PER_SHEET = 150, 267, 10, 50
FRAME_RE = re.compile(r"^(.*)_f(\d+)$")
OUTLIER_SCORE = 0.01   # IoU(prev, next) minus mean IoU(frame, neighbours) above this -> listed
N_OUTLIERS_SHOWN = 30


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else 1.0


def temporal_outliers(ds: Path, stems: list[str]) -> list[tuple[float, str]]:
    """(score, stem) for frames whose neighbours (frame +-1 of the same video) agree better without them."""
    frames = {}
    for s in stems:
        m = FRAME_RE.match(s)
        if m:
            frames[(m.group(1), int(m.group(2)))] = s
    small = {}

    def mask(stem: str) -> np.ndarray:
        if stem not in small:
            m = cv2.imread(str(ds / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            small[stem] = cv2.resize(m, (m.shape[1] // 4, m.shape[0] // 4), interpolation=cv2.INTER_AREA) >= 128
        return small[stem]

    scored = []
    for (video, n), stem in sorted(frames.items()):
        prev, nxt = frames.get((video, n - 1)), frames.get((video, n + 1))
        if prev is None or nxt is None:
            continue
        a, b, c = mask(prev), mask(stem), mask(nxt)
        scored.append((iou(a, c) - (iou(a, b) + iou(b, c)) / 2, stem))
    return sorted(scored, reverse=True)

def load_excluded(dataset: Path) -> set[str]:
    path = dataset / "exclude.txt"
    if not path.exists():
        return set()
    stems = (line.split("#")[0].strip() for line in path.read_text().splitlines())
    return {s for s in stems if s}


def tile(dataset: Path, row: dict, i: int, excluded: bool) -> np.ndarray:
    im = cv2.resize(cv2.imread(str(dataset / "overlays" / f"{row['stem']}.jpg")), (TILE_W, TILE_H))
    colour = (0, 0, 255) if excluded else (0, 200, 255) if row["flags"] else (0, 255, 0)
    text = "EXCLUDED" if excluded else row["flags"]
    cv2.putText(im, f"{i} {text}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
    return im


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-validate")
    p.add_argument("dataset", type=Path)
    args = p.parse_args()
    ds = args.dataset
    rows = list(csv.DictReader((ds / "index.csv").open()))

    errors = []
    excluded = load_excluded(ds)
    for r in rows:
        img = cv2.imread(str(ds / "images" / f"{r['stem']}.jpg"))
        mask = cv2.imread(str(ds / "masks" / f"{r['stem']}.png"), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            errors.append(f"{r['stem']}: missing image or mask")
            continue
        if mask.ndim != 2 or img.shape[:2] != mask.shape:
            errors.append(f"{r['stem']}: image {img.shape} vs mask {mask.shape}")
        if not set(np.unique(mask)) <= {0, 255}:
            errors.append(f"{r['stem']}: mask is not binary 0/255")
        no_hand = "no_hand" in r["flags"].split("|")
        if no_hand and mask.any():
            errors.append(f"{r['stem']}: flagged no_hand but mask is not empty")
        if not no_hand and not mask.any() and r["stem"] not in excluded:
            errors.append(f"{r['stem']}: empty mask")
    stems = {r["stem"] for r in rows}
    errors += [f"exclude.txt: {e} not in index.csv" for e in sorted(excluded - stems)]
    for sub, ext in (("images", "jpg"), ("masks", "png")):
        orphans = {f.stem for f in (ds / sub).glob(f"*.{ext}")} - stems
        errors += [f"{sub}/{o}.{ext}: not in index.csv" for o in sorted(orphans)]

    flags = collections.Counter(f for r in rows for f in (r["flags"].split("|") if r["flags"] else ["ok"]))
    areas = np.array([float(r["area"]) for r in rows if "no_hand" not in r["flags"].split("|")])
    n_neg = sum("no_hand" in r["flags"].split("|") for r in rows)
    print(f"{len(rows)} samples ({len(rows) - n_neg} hand, {n_neg} no-hand), models {sorted({r['model'] for r in rows})}")
    print(f"flags: {dict(flags)}")
    print(f"mask area: min {areas.min():.3f} median {np.median(areas):.3f} max {areas.max():.3f}")
    print(f"excluded after review: {len(excluded)} -> {len(rows) - len(excluded & stems)} usable samples")

    review = ds / "review"
    review.mkdir(exist_ok=True)
    for old in review.glob("sheet_*.jpg"):
        old.unlink()
    for s in range(0, len(rows), PER_SHEET):
        tiles = [tile(ds, r, i, r["stem"] in excluded) for i, r in enumerate(rows[s:s + PER_SHEET], s)]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
        sheet = np.vstack([np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)])
        cv2.imwrite(str(review / f"sheet_{s // PER_SHEET:02d}.jpg"), sheet)
    print(f"contact sheets: {review}/")

    outliers = temporal_outliers(ds, [r["stem"] for r in rows])
    if outliers:
        flagged = [(sc, st) for sc, st in outliers if sc > OUTLIER_SCORE]
        print(f"temporal check: {len(outliers)} frames scored, {len(flagged)} above {OUTLIER_SCORE} "
              f"(worst {outliers[0][0]:.3f} {outliers[0][1]})")
        index_of = {r["stem"]: i for i, r in enumerate(rows)}
        shown = outliers[:N_OUTLIERS_SHOWN]
        for sc, st in shown[:10]:
            print(f"   {sc:.3f}  #{index_of[st]}  {st}{'  (excluded)' if st in excluded else ''}")
        tiles = [tile(ds, rows[index_of[st]], index_of[st], st in excluded) for _, st in shown]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
        cv2.imwrite(str(review / "outliers.jpg"),
                    np.vstack([np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)]))

    if errors:
        print(f"\n{len(errors)} ERRORS")
        print("\n".join(errors[:50]))
        sys.exit(1)
    print("\nDATASET OK")


if __name__ == "__main__":
    main()
