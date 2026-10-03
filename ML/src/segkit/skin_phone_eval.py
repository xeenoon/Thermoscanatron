"""Score skin models the way the phone sees: the centred box of each frame, frame after frame.

    segkit-skin-phone-eval data/skin_v2 --stems-from hand_20261003_194033 hand_20261003_194728 \\
        --model runs/skin_big_v2/best.pt:mobilenetv3_large_100:384 --model runs/skin_small_v2/best.pt:mobilenetv3_small_100:256:prev

For each model: skin IoU, and on frames with a labelled hand the hand IoU and hand recall (share of the hand's
pixels found: a missed thumb or fingertip shows up here), plus no-skin frames wrongly flagged. A model with a
previous-mask input (":prev") is run twice: with an empty previous mask (worst case, first frame) and fed its own
prediction for the previous sampled frame of the same video (how the phone runs it). Only validation-block frames
are scored (never trained on), unless --all-frames. Writes worst.jpg per model:
the frames with the lowest hand recall, label green, prediction red.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import is_val, normalize
from segkit.models.handseg import HandSegNet

ZONE = 9


def phone_crop(img: np.ndarray, size: int, interp=cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    side = int(0.9 * min(h, w))
    x0, y0 = (w - side) // 2, (h - side) // 2
    return cv2.resize(img[y0:y0 + side, x0:x0 + side], (size, size), interpolation=interp)


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-skin-phone-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--stems-from", nargs="+", required=True, help="video stems (frames <video>_fNNNNNN)")
    ap.add_argument("--model", action="append", required=True, help="path:encoder:size[:prev]")
    ap.add_argument("--out", type=Path, default=Path("runs/phone_eval"))
    ap.add_argument("--all-frames", action="store_true",
                    help="score every frame, not only validation blocks (only fair for models that never trained on them)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader((args.dataset / "index.csv").open())
            if any(r["stem"].startswith(v + "_f") for v in args.stems_from)
            and (args.all_frames or is_val(r["stem"]))
            and "person_no_skin" not in r["flags"].split("|")]
    ex = args.dataset / "exclude.txt"
    if ex.exists():
        bad = {l.split("#")[0].strip() for l in ex.read_text().splitlines()}
        rows = [r for r in rows if r["stem"] not in bad]
    rows.sort(key=lambda r: r["stem"])
    frames = []
    for r in rows:
        s = r["stem"]
        rgb = cv2.cvtColor(cv2.imread(str(args.dataset / "images" / f"{s}.jpg")), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(args.dataset / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        hd = cv2.imread(str(args.dataset / "hands" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        frames.append((s, phone_crop(rgb, 384), phone_crop(m, 384, cv2.INTER_NEAREST) > 0,
                       phone_crop(hd if hd is not None else np.zeros_like(m), 384, cv2.INTER_NEAREST) > 0))
    print(f"{len(frames)} frames from {args.stems_from}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for spec in args.model:
        parts = spec.split(":")
        path, enc, size = parts[0], parts[1], int(parts[2])
        prev = len(parts) > 3 and parts[3] == "prev"
        net = HandSegNet(enc, in_chans=4 if prev else 3).to(device).eval()
        state = torch.load(path, map_location=device)
        net.load_state_dict({k: v for k, v in state.items() if not k.startswith("aux.")}, strict=False)
        for mode in (["empty", "chained"] if prev else ["-"]):
            res = {"skin": [], "hand_iou": [], "hand_rec": [], "neg_fp": []}
            worst = []
            last = None
            last_video = None
            for s, rgb, gt, hand in frames:
                video = s.rsplit("_f", 1)[0]
                x = normalize(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
                if prev:
                    pm = np.zeros((size, size), np.float32)
                    if mode == "chained" and last is not None and video == last_video:
                        pm = last
                    x = torch.cat([x, torch.from_numpy(pm)[None]])
                with torch.no_grad():
                    lo, pr = net(x[None].to(device))
                small = (lo[0, 0] > 0).float().cpu().numpy()
                last, last_video = small, video
                pred = cv2.resize(small, (384, 384), interpolation=cv2.INTER_NEAREST) > 0.5
                present = float(torch.sigmoid(pr[0, 0]))
                if gt.mean() < 0.002:
                    res["neg_fp"].append(float(pred.mean() > 0.005 or present > 0.5))
                    continue
                res["skin"].append((pred & gt).sum() / (pred | gt).sum())
                if hand.sum() > 400:
                    zone = cv2.dilate(hand.astype(np.uint8), np.ones((2 * ZONE + 1,) * 2, np.uint8)) > 0
                    pz = pred & zone
                    res["hand_iou"].append((pz & hand).sum() / (pz | hand).sum())
                    rec = (pz & hand).sum() / hand.sum()
                    res["hand_rec"].append(rec)
                    worst.append((rec, s, rgb, hand, pred))
            name = f"{Path(path).parent.name}{'' if mode == '-' else ' prev=' + mode}"
            print(f"{name:40s} skin IoU {np.mean(res['skin']):.3f} ({len(res['skin'])})  hand IoU "
                  f"{np.mean(res['hand_iou']):.3f}  hand recall {np.mean(res['hand_rec']):.3f} "
                  f"(p10 {np.percentile(res['hand_rec'], 10):.3f}, n={len(res['hand_rec'])})  "
                  f"no-skin flagged {int(sum(res['neg_fp']))}/{len(res['neg_fp'])}", flush=True)
            worst.sort(key=lambda t: t[0])
            tiles = []
            for rec, s, rgb, hand, pred in worst[:16]:
                v = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                for mm, c in ((hand, (0, 255, 0)), (pred, (0, 0, 255))):
                    cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(v, cs, -1, c, 2)
                v = cv2.resize(v, (240, 240))
                cv2.putText(v, f"{rec:.2f} {s[-14:]}", (3, 16), 0, 0.45, (0, 255, 255), 1)
                tiles.append(v)
            while len(tiles) % 8:
                tiles.append(np.zeros((240, 240, 3), np.uint8))
            if tiles:
                cv2.imwrite(str(args.out / f"worst_{name.replace(' ', '_').replace('=', '-')}.jpg"),
                            np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


if __name__ == "__main__":
    main()
