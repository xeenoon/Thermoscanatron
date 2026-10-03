"""Cross-check auto-labels against a trained model, and rank frames for manual review.

    segkit-audit data/hands_new --model runs/handseg_v3/best.pt

The auto-labeller (MediaPipe + rembg) fails in known ways: side-on, dark or blurred hands become "no hand",
and a face, a body or a painting can get labelled as a hand. A model trained on reviewed data catches most of
these: each frame is run through it (the training pipeline's validation crop) and compared with its label:
  missed_hand   label says no hand, the model is confident there is one   (MediaPipe miss)
  false_hand    label says hand, the model is confident there is none     (face / painting / arm)
  outline       both say hand but the outlines disagree (IoU < OUTLINE_IOU)
Writes <dataset>/audit.csv (every frame, sorted most suspicious first) and review/audit_NN.jpg sheets of the
suspicious ones (label outline green, model outline red). Rejected stems go into <dataset>/exclude.txt as
usual (segkit-validate, training both honour it).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import HandCrops
from segkit.models.handseg import HandSegNet

OUTLINE_IOU = 0.7
CONFIDENT = 0.8
TILE = 200
COLS, ROWS = 8, 6


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-audit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--model", type=Path, required=True, help="HandSegNet state dict (best.pt)")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    rows = list(csv.DictReader((args.dataset / "index.csv").open()))
    stems = [r["stem"] for r in rows]
    flags = {r["stem"]: r["flags"] for r in rows}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HandSegNet().to(device).eval()
    net.load_state_dict(torch.load(args.model, map_location=device))
    ds = HandCrops(args.dataset, stems, args.size, train=False)

    out = []
    with torch.no_grad():
        for b0 in range(0, len(ds), args.batch):
            items = [ds[i] for i in range(b0, min(b0 + args.batch, len(ds)))]
            x = torch.stack([it[0] for it in items]).to(device)
            mask_logits, present_logit = net(x)
            pm = (mask_logits[:, 0] > 0).cpu().numpy()
            pp = torch.sigmoid(present_logit[:, 0]).cpu().numpy()
            for k, (img, m, present) in enumerate(items):
                i = b0 + k
                lm = m[0].numpy() > 0.5
                label_hand = bool(present[0] > 0.5)
                union = (lm | pm[k]).sum()
                iou = float((lm & pm[k]).sum() / union) if union else 1.0
                if not label_hand and pp[k] > CONFIDENT:
                    kind, score = "missed_hand", float(pp[k])
                elif label_hand and pp[k] < 1 - CONFIDENT:
                    kind, score = "false_hand", float(1 - pp[k])
                elif label_hand and iou < OUTLINE_IOU:
                    kind, score = "outline", 1 - iou
                else:
                    kind, score = "", 0.0
                row = {"stem": stems[i], "kind": kind, "score": f"{score:.3f}", "model_present": f"{pp[k]:.3f}",
                       "label_hand": int(label_hand), "iou": f"{iou:.3f}", "flags": flags[stems[i]], "_i": i}
                if kind:   # keep the model's mask only for frames that go on a review sheet
                    row["_pred"] = np.packbits(pm[k])
                out.append(row)
            print(f"{min(b0 + args.batch, len(ds))}/{len(ds)}", end="\r", flush=True)
    out.sort(key=lambda r: -float(r["score"]))
    fields = ["stem", "kind", "score", "model_present", "label_hand", "iou", "flags"]
    with (args.dataset / "audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    suspects = [r for r in out if r["kind"]]
    counts = {k: sum(r["kind"] == k for r in suspects) for k in ("missed_hand", "false_hand", "outline")}
    print(f"\n{len(out)} frames; suspicious: {counts}")
    write_sheets(args.dataset, suspects, ds, args.size)


def write_sheets(dataset: Path, suspects: list[dict], ds: HandCrops, size: int) -> None:
    from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD
    (dataset / "review").mkdir(exist_ok=True)
    per = COLS * ROWS
    for s in range(0, len(suspects), per):
        tiles = []
        for j, r in enumerate(suspects[s:s + per]):
            img, m_label, _ = ds[r["_i"]]
            rgb = ((img.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            pred = np.unpackbits(r["_pred"])[:size * size].reshape(size, size).astype(bool)
            for m, col in ((m_label[0].numpy() > 0.5, (0, 255, 0)), (pred, (0, 0, 255))):
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cs, -1, col, 2)
            vis = cv2.resize(vis, (TILE, TILE))
            cv2.putText(vis, f"{s + j} {r['kind']} {float(r['model_present']):.2f}", (3, 14), 0, 0.42, (0, 255, 255), 1)
            cv2.putText(vis, r["stem"][-24:], (3, TILE - 6), 0, 0.35, (255, 255, 255), 1)
            tiles.append(vis)
        while len(tiles) % COLS:
            tiles.append(np.zeros((TILE, TILE, 3), np.uint8))
        sheet = np.vstack([np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)])
        cv2.imwrite(str(dataset / "review" / f"audit_{s // per:02d}.jpg"), sheet)


if __name__ == "__main__":
    main()
