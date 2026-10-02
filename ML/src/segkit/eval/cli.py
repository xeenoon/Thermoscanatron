"""segkit-eval check <labels>          render label overlays to eyeball annotation quality
   segkit-eval score <labels> <preds>  score predicted masks (<stem>.png, nonzero = hand)"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from segkit.eval.labels import load_dir
from segkit.eval.metrics import F_THRESHOLDS_PX, boundary, score

TARGET_P95_PX = 2.0


def check(label_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_dir(label_dir)
    if not labels:
        sys.exit(f"no *.json labels in {label_dir}")
    for lab in labels:
        img = cv2.imread(str(lab.image_path))
        if img is None:
            sys.exit(f"{lab.stem}: cannot read image {lab.image_path}")
        if img.shape[:2] != lab.hand.shape:
            sys.exit(f"{lab.stem}: image {img.shape[:2]} != label {lab.hand.shape}")
        vis = img.copy()
        vis[lab.hand] = (0.6 * vis[lab.hand] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
        vis[lab.ignore] = (0.6 * vis[lab.ignore] + 0.4 * np.array([0, 0, 255])).astype(np.uint8)
        vis[boundary(lab.hand)] = (0, 255, 255)
        cv2.imwrite(str(out_dir / f"{lab.stem}.jpg"), vis)
        print(f"{lab.stem:30s} {img.shape[1]}x{img.shape[0]}  hand {lab.hand.mean() * 100:5.1f}%  "
              f"ignore {lab.ignore.mean() * 100:4.1f}%")
    print(f"\n{len(labels)} labels OK, overlays in {out_dir}")


def score_dir(label_dir: Path, pred_dir: Path, csv_path: Path | None) -> None:
    labels = load_dir(label_dir)
    if not labels:
        sys.exit(f"no *.json labels in {label_dir}")
    rows = []
    for lab in labels:
        pred_path = pred_dir / f"{lab.stem}.png"
        pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            print(f"{lab.stem:30s} MISSING prediction, scored as empty")
            pred = np.zeros(lab.hand.shape, np.uint8)
        if pred.shape != lab.hand.shape:
            sys.exit(f"{lab.stem}: prediction {pred.shape} != label {lab.hand.shape}")
        s = score(pred > 0, lab.hand, lab.ignore)
        rows.append({"image": lab.stem, "iou": s.iou, "p50_px": s.p50_px, "p95_px": s.p95_px,
                     "max_px": s.max_px, **{f"f@{t}px": s.f_at[t] for t in F_THRESHOLDS_PX}})

    keys = [k for k in rows[0] if k != "image"]
    print(f"{'image':30s} " + " ".join(f"{k:>8s}" for k in keys))
    for r in rows:
        print(f"{r['image']:30s} " + " ".join(f"{r[k]:8.3f}" for k in keys))
    p95s = np.array([r["p95_px"] for r in rows])
    print(f"\nimages {len(rows)}  mean IoU {np.mean([r['iou'] for r in rows]):.3f}  "
          f"mean F@2px {np.mean([r['f@2px'] for r in rows]):.3f}  "
          f"median p95 {np.median(p95s):.2f}px  worst p95 {p95s.max():.2f}px ({rows[p95s.argmax()]['image']})")
    print(f"target p95 <= {TARGET_P95_PX}px: {(p95s <= TARGET_P95_PX).sum()}/{len(rows)} images pass")

    if csv_path:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-eval")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("labels", type=Path)
    c.add_argument("--out", type=Path, default=Path("runs/eval_check"))
    s = sub.add_parser("score")
    s.add_argument("labels", type=Path)
    s.add_argument("preds", type=Path)
    s.add_argument("--csv", type=Path)
    args = p.parse_args()
    if args.cmd == "check":
        check(args.labels, args.out)
    else:
        score_dir(args.labels, args.preds, args.csv)


if __name__ == "__main__":
    main()
