"""Evaluation sheet for a trained HandSegNet checkpoint.

segkit-report data/hands_all runs/handseg_v2/best.pt --out runs/handseg_v2/eval_report.jpg

Rows: validation frames spread over the set, the worst frames by p95 boundary error, validation
background crops (should say no hand), and unseen images (assets/*.jpg; prediction only).
Green = label, red = prediction; the tile caption shows p95 error and the hand-present score.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD, HandCrops, crop_affine, load_split, normalize
from segkit.eval.metrics import score
from segkit.models.handseg import HandSegNet

TILE = 360
COLS = 5


def tile(x: torch.Tensor, gt: np.ndarray | None, pred: np.ndarray, caption: str, ok: bool) -> np.ndarray:
    rgb = ((x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for mask, colour in ((gt, (0, 255, 0)), (pred, (0, 0, 255))):
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(bgr, contours, -1, colour, 1)
    bgr = cv2.resize(bgr, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(bgr, (0, 0), (TILE, 26), (0, 0, 0), -1)
    cv2.putText(bgr, caption, (5, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255) if ok else (0, 0, 255), 1)
    return bgr


def grid(tiles: list[np.ndarray]) -> np.ndarray:
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
    return np.vstack([np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)])


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-report")
    p.add_argument("dataset", type=Path)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--unseen", type=Path, nargs="*", default=sorted(Path("assets").glob("*.jpg")))
    args = p.parse_args()
    out = args.out or args.checkpoint.with_name("eval_report.jpg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = HandSegNet()
    net.load_state_dict(torch.load(args.checkpoint, map_location=device))
    net = net.to(device).eval()

    def run(x: torch.Tensor) -> tuple[np.ndarray, float]:
        mask_logits, present_logit = net(x[None].to(device))
        return (mask_logits[0, 0] > 0).cpu().numpy(), torch.sigmoid(present_logit).item()

    _, val = load_split(args.dataset)
    hands = HandCrops(args.dataset, val, train=False)
    results = []
    for i in range(len(hands)):
        x, m, present = hands[i]
        pred, prob = run(x)
        gt = m[0].numpy() > 0.5
        if present.item():
            results.append((score(pred, gt).p95_px, i, x, gt, pred, prob))

    p95 = np.array([r[0] for r in results])
    print(f"hand frames {len(results)}: p95 median {np.median(p95):.2f}px, 90th pct {np.percentile(p95, 90):.2f}px, "
          f"<=2px {np.mean(p95 <= 2) * 100:.1f}%")

    def caption(r):
        return f"{hands.stems[r[1]][-14:]} p95 {r[0]:.1f}px hand {r[5]:.2f}"

    spread = [results[k] for k in np.linspace(0, len(results) - 1, COLS * 2).astype(int)]
    worst = [results[k] for k in np.argsort(p95)[-COLS:]]
    rows = [tile(r[2], r[3], r[4], caption(r), r[0] <= 2 and r[5] > 0.5) for r in spread + worst]

    background = HandCrops(args.dataset, val[::2], train=False, background_only=True)
    bg_scores = []
    for i in range(len(background)):
        x, m, present = background[i]
        if not present.item():
            pred, prob = run(x)
            bg_scores.append((prob, pred.mean(), x, pred))
    false_pos = [b for b in bg_scores if b[0] > 0.5 or b[1] >= 0.005]
    print(f"background crops {len(bg_scores)}: false positives {len(false_pos)} "
          f"({len(false_pos) / max(1, len(bg_scores)) * 100:.1f}%)")
    shown = sorted(bg_scores, key=lambda b: -b[0])[:COLS]  # hardest negatives first
    rows += [tile(b[2], None, b[3], f"NO-HAND crop: hand {b[0]:.2f}", b[0] <= 0.5 and b[1] < 0.005) for b in shown]

    for path in args.unseen:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        side = min(h, w)
        for cx, cy in ((w / 2, h / 2), (w / 2, h * 0.75), (w / 2, h * 0.3)):
            mtx = crop_affine((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), 384, 1.0, (0, 0), 0)
            x = normalize(cv2.warpAffine(rgb, mtx, (384, 384), borderMode=cv2.BORDER_CONSTANT))
            pred, prob = run(x)
            rows.append(tile(x, None, pred, f"UNSEEN {path.stem}: hand {prob:.2f}", True))

    cv2.imwrite(str(out), grid(rows))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
