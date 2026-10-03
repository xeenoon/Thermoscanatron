"""Skin model fairness report: accuracy per person on held-out frames, and the same frames at six skin tones.

    segkit-skin-eval data/skin_v1 --model runs/skin_v1/best.pt --people data/skin_v1/people.json \\
        [--model-size 384] --out runs/skin_v1/fairness

people.json: {"<stem>": "<person>", ...} for held-out (validation-block) frames, tagged by hand. A frame with
several people gets the tag of the person filling most of it.

Per person: mean IoU of the skin mask, recall (share of labelled skin found) and precision (share of predicted
skin that is skin), on the validation crop. Plus the false-positive rate on held-out frames with no skin.
Tone test: every held-out skin frame is recoloured to each of segkit.skin_tone.TONES (the labelled skin only;
texture and shading kept) and scored again; the table shows IoU per tone, overall and per person, and the
worst-tone drop relative to the original frame. Writes report.md, results.csv and tones.jpg (examples).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD, HandCrops, normalize
from segkit.models.handseg import HandSegNet
from segkit.skin_tone import TONES, recolour


def scores(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    inter = float((pred & gt).sum())
    union = float((pred | gt).sum())
    return (inter / union if union else 1.0, inter / gt.sum() if gt.sum() else 1.0,
            inter / pred.sum() if pred.sum() else (1.0 if not gt.sum() else 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-skin-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-size", type=int, default=384, help="the model's input size (small model: 256)")
    ap.add_argument("--people", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    people = json.loads(args.people.read_text())
    rows = list(csv.DictReader((args.dataset / "index.csv").open()))
    neg = [r["stem"] for r in rows if "no_skin" in r["flags"].split("|") and r["stem"] in people]
    pos = [r["stem"] for r in rows if "no_skin" not in r["flags"].split("|") and r["stem"] in people]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HandSegNet().to(device).eval()
    net.load_state_dict(torch.load(args.model, map_location=device))
    ds = HandCrops(args.dataset, pos + neg, 384, train=False)
    s = args.model_size

    @torch.no_grad()
    def run(rgb: np.ndarray) -> tuple[np.ndarray, float]:
        x = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_AREA) if s != rgb.shape[0] else rgb
        m, p = net(normalize(x)[None].to(device))
        m = (m[0, 0] > 0).cpu().numpy().astype(np.uint8)
        if s != rgb.shape[0]:
            m = cv2.resize(m, rgb.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        return m.astype(bool), float(torch.sigmoid(p[0, 0]))

    res = []
    examples = []
    for i, stem in enumerate(pos + neg):
        x, m, _ = ds[i]
        rgb = ((x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
        gt = m[0].numpy() > 0.5
        pred, pres = run(rgb)
        row = {"stem": stem, "person": people[stem], "skin": int(gt.any()), "present": f"{pres:.3f}"}
        if gt.any():
            iou, rec, prec = scores(pred, gt)
            row.update(iou=f"{iou:.4f}", recall=f"{rec:.4f}", precision=f"{prec:.4f}")
            tiles = [] if len(examples) < 6 and i % 7 == 0 else None
            for name, L, a, b in TONES:
                rc = recolour(rgb, gt, L, a, b)
                p2, _ = run(rc)
                row[f"iou_{name}"] = f"{scores(p2, gt)[0]:.4f}"
                if tiles is not None:
                    vis = cv2.cvtColor(rc, cv2.COLOR_RGB2BGR)
                    cs, _ = cv2.findContours(p2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(vis, cs, -1, (0, 255, 0), 2)
                    vis = cv2.resize(vis, (192, 192))
                    cv2.putText(vis, f"{name} {float(row[f'iou_{name}']):.2f}", (4, 18), 0, 0.55, (0, 255, 255), 2)
                    tiles.append(vis)
            if tiles is not None:
                examples.append(np.hstack(tiles))
        else:
            row["fp"] = int(pred.mean() > 0.005 or pres > 0.5)
        res.append(row)
        print(f"{i + 1}/{len(pos) + len(neg)}", end="\r", flush=True)

    fields = ["stem", "person", "skin", "present", "iou", "recall", "precision", "fp"] + [f"iou_{t[0]}" for t in TONES]
    with (args.out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(res)
    if examples:
        cv2.imwrite(str(args.out / "tones.jpg"), np.vstack(examples))

    by = defaultdict(list)
    for r in res:
        if r["skin"]:
            by[r["person"]].append(r)
    mean = lambda rs, k: float(np.mean([float(r[k]) for r in rs]))
    lines = [f"# Skin model fairness report\n\nModel `{args.model}` ({s} px), dataset `{args.dataset}`, "
             f"{len(pos)} held-out frames with skin, {len(neg)} without.\n",
             "## Accuracy per person (held-out frames)\n",
             "| Person | Frames | IoU | Recall | Precision |", "|---|---|---|---|---|"]
    for p in sorted(by):
        rs = by[p]
        lines.append(f"| {p} | {len(rs)} | {mean(rs, 'iou'):.3f} | {mean(rs, 'recall'):.3f} | {mean(rs, 'precision'):.3f} |")
    allp = [r for r in res if r["skin"]]
    lines.append(f"| **all** | {len(allp)} | {mean(allp, 'iou'):.3f} | {mean(allp, 'recall'):.3f} | "
                 f"{mean(allp, 'precision'):.3f} |")
    negs = [r for r in res if not r["skin"]]
    if negs:
        lines.append(f"\nNo-skin frames flagged as skin: {sum(r['fp'] for r in negs)}/{len(negs)}.\n")
    lines += ["## Same frames, recoloured skin (IoU)\n",
              "| Person | original | " + " | ".join(t[0] for t in TONES) + " | worst drop |",
              "|---|---|" + "---|" * len(TONES) + "---|"]
    for p in sorted(by) + ["all"]:
        rs = allp if p == "all" else by[p]
        tones = [mean(rs, f"iou_{t[0]}") for t in TONES]
        orig = mean(rs, "iou")
        lines.append(f"| {p} | {orig:.3f} | " + " | ".join(f"{v:.3f}" for v in tones) +
                     f" | {orig - min(tones):+.3f} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
