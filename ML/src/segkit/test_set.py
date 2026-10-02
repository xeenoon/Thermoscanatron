"""Score a checkpoint on a held-out test set exactly the way the phone demo runs it.

segkit-test data/test_hallway runs/handseg_v3/best.pt

The test dir is a segkit-label output whose images are already the network's input crops (e.g. the
phone's diagnostics dumps). Each image is resized to the model size and run whole; like the app, the
mask is reduced to its largest blob and "hand" means the hand-present score > 0.5.
Labels: frames flagged no_hand are negatives; frames flagged landmarks_outside or listed in exclude.txt
are skipped (label not trusted).
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import normalize
from segkit.eval.metrics import score
from segkit.models.handseg import HandSegNet

SIZE = 384


def largest_blob(mask: np.ndarray) -> np.ndarray:
    n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    if n <= 1:
        return np.zeros_like(mask, bool)
    return cc == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


@torch.no_grad()
def evaluate(test_dir: Path, checkpoint: Path, device: str = "cpu") -> dict:
    net = HandSegNet()
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net = net.to(device).eval()
    rows = list(csv.DictReader((test_dir / "index.csv").open()))
    exclude_path = test_dir / "exclude.txt"
    excluded = set()
    if exclude_path.exists():
        excluded = {l.split("#")[0].strip() for l in exclude_path.read_text().splitlines()} - {""}

    tp = fn = fp = tn = 0
    ious, f2s = [], []
    for r in rows:
        flags = set(r["flags"].split("|"))
        if "landmarks_outside" in flags or r["stem"] in excluded:
            continue
        rgb = cv2.cvtColor(cv2.imread(str(test_dir / "images" / f"{r['stem']}.jpg")), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(test_dir / "masks" / f"{r['stem']}.png"), cv2.IMREAD_GRAYSCALE) > 0
        if rgb.shape[:2] != (SIZE, SIZE):
            rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt.astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
        mask_logits, present_logit = net(normalize(rgb)[None].to(device))
        said_hand = torch.sigmoid(present_logit).item() > 0.5
        is_hand = "no_hand" not in flags
        tp += is_hand and said_hand
        fn += is_hand and not said_hand
        fp += not is_hand and said_hand
        tn += not is_hand and not said_hand
        if is_hand:
            s = score(largest_blob((mask_logits[0, 0] > 0).cpu().numpy()), gt)
            ious.append(s.iou)
            f2s.append(s.f_at[2])
    n_hand, n_none = tp + fn, fp + tn
    return {"hand_frames": n_hand, "no_hand_frames": n_none,
            "hand_recall": tp / n_hand if n_hand else float("nan"),
            "false_alarm_rate": fp / n_none if n_none else float("nan"),
            "iou": float(np.mean(ious)) if ious else float("nan"),
            "f@2px": float(np.mean(f2s)) if f2s else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-test")
    p.add_argument("test_dir", type=Path)
    p.add_argument("checkpoints", type=Path, nargs="+")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt in args.checkpoints:
        m = evaluate(args.test_dir, ckpt, device)
        print(f"{ckpt}: hand frames {m['hand_frames']} -> said HAND {m['hand_recall'] * 100:.1f}%  |  "
              f"no-hand frames {m['no_hand_frames']} -> false alarm {m['false_alarm_rate'] * 100:.1f}%  |  "
              f"outline IoU {m['iou']:.3f}  F@2px {m['f@2px']:.3f}")


if __name__ == "__main__":
    main()
