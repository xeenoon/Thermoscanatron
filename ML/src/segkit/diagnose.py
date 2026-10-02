"""Explain the demo app's "no hand" mistakes from a phone diagnostics session.

adb pull /sdcard/Android/data/com.euhack.hello/files/diagnostics data/
segkit-diagnose data/diagnostics/session_<time> --run runs/handseg_v2

For each dumped NO-HAND frame (the exact 384x384 network input saved by the phone):
  1. parity: re-run the same crop through the .pte (ExecuTorch) and best.pt (PyTorch) here and compare
     the hand-present score with what the phone logged; for the float dumps, compare the phone's
     normalised tensor with Python's normalize() (catches RGB/BGR, mean/std, layout bugs);
  2. ground truth: MediaPipe on the crop says whether a hand is really there, and where;
  3. which head failed: did the mask still find the hand while the hand-present head said no;
  4. distribution shift: hand size / cut-off in the crop vs the training crops.
Writes <session>/diagnosis.jpg (contact sheet) and diagnosis.csv, and prints a summary.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import HandCrops, IMAGENET_MEAN, IMAGENET_STD, load_split, normalize
from segkit.export import PteRunner
from segkit.hand_landmarks import HandLandmarks
from segkit.models.handseg import HandSegNet, ProbabilityHead

SIZE = 384
MASK_FOUND_FRACTION = 0.005  # same "hand present" area rule as training
TILE = 256
COLS = 6


def hand_geometry(landmarks: list[np.ndarray]) -> dict:
    """Size of the hand (landmark bbox, longest side / crop side) and whether it runs off the crop."""
    pts = np.concatenate(landmarks)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    margin = 4
    return {"hand_size": float(max(x1 - x0, y1 - y0) / SIZE),
            "cut_off": bool(x0 < margin or y0 < margin or x1 > SIZE - margin or y1 > SIZE - margin),
            "centre_offset": float(np.hypot((x0 + x1) / 2 - SIZE / 2, (y0 + y1) / 2 - SIZE / 2) / SIZE)}


def training_hand_sizes(dataset: Path, detector: HandLandmarks, n: int = 150) -> np.ndarray:
    """Hand size (same measure) in random training crops, to compare against the phone's crops."""
    train, _ = load_split(dataset)
    crops = HandCrops(dataset, train, SIZE, train=True)
    rng = np.random.default_rng(0)
    sizes = []
    for i in rng.choice(len(train), size=min(n, len(train)), replace=False):
        x, _, present = crops[int(i)]
        if not present.item():
            continue
        rgb = ((x.numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
        hands = detector.detect(rgb)
        if hands:
            sizes.append(hand_geometry(hands)["hand_size"])
    return np.array(sizes)


def tile(rgb: np.ndarray, mask: np.ndarray, landmarks: list[np.ndarray], caption: list[str], bad: bool) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), 2)
    for hand in landmarks:
        for x, y in hand:
            cv2.circle(bgr, (int(x), int(y)), 3, (255, 0, 255), -1)
    bgr = cv2.resize(bgr, (TILE, TILE))
    cv2.rectangle(bgr, (0, 0), (TILE, 16 * len(caption) + 4), (0, 0, 0), -1)
    for k, line in enumerate(caption):
        cv2.putText(bgr, line, (4, 14 + 16 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255) if bad else (0, 255, 255), 1)
    return bgr


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-diagnose")
    p.add_argument("session", type=Path)
    p.add_argument("--run", type=Path, default=Path("runs/handseg_v2"), help="dir with best.pt and handseg.pte")
    p.add_argument("--dataset", type=Path, default=Path("data/hands_all"),
                   help="training dataset, for the hand-size comparison (skipped if missing)")
    args = p.parse_args()
    s = args.session

    phone = [json.loads(line) for line in (s / "nohand.jsonl").read_text().splitlines() if line.strip()]
    frames = list(csv.DictReader((s / "frames.csv").open()))
    print(f"session {s.name}: {len(frames)} frames logged, {len(phone)} NO-HAND dumps")

    present_all = np.array([float(f["present"]) for f in frames])
    no_hand = present_all <= 0.5
    runs, cur = [], 0
    for v in no_hand:
        cur = cur + 1 if v else 0
        if cur == 1:
            runs.append(0)
        if v:
            runs[-1] = cur
    print(f"frames called NO HAND: {no_hand.mean() * 100:.1f}%  "
          f"(streaks: {len(runs)}, median {np.median(runs) if runs else 0:.0f} frames, longest {max(runs, default=0)})")

    # 1. Preprocessing parity on the float dumps.
    for f32 in sorted(s.glob("*_input_f32.bin")):
        phone_x = np.fromfile(f32, dtype="<f4").reshape(3, SIZE, SIZE)
        rgb = cv2.cvtColor(cv2.imread(str(f32.with_name(f32.name.replace("_input_f32.bin", "_input.png")))),
                           cv2.COLOR_BGR2RGB)
        ours = normalize(rgb).numpy()
        swapped = normalize(rgb[:, :, ::-1].copy()).numpy()
        print(f"preprocessing {f32.name}: max |phone - python| = {np.abs(phone_x - ours).max():.2e}"
              f"  (vs BGR-swapped {np.abs(phone_x - swapped).max():.2e})")

    pte = PteRunner(args.run / "handseg.pte")
    net = HandSegNet()
    net.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu"))
    torch_model = ProbabilityHead(net).eval()
    detector = HandLandmarks()

    rows, tiles = [], []
    for d in phone:
        name = f"{d['dump']:05d}"
        rgb = cv2.cvtColor(cv2.imread(str(s / f"{name}_input.png")), cv2.COLOR_BGR2RGB)
        x = normalize(rgb)[None]
        mask_pte, present_pte = (t.numpy() for t in pte.method.execute([x]))
        mask_t, present_t = (t.numpy() for t in torch_model(x))
        hands = detector.detect(rgb)
        geo = hand_geometry(hands) if hands else {"hand_size": np.nan, "cut_off": False, "centre_offset": np.nan}
        mask = mask_pte[0, 0]
        mask_frac = float((mask > 0.5).mean())
        lm_on_mask = float(np.mean([mask[min(SIZE - 1, max(0, int(y))), min(SIZE - 1, max(0, int(x_)))] > 0.5
                                    for hand in hands for x_, y in hand])) if hands else np.nan
        row = {"dump": d["dump"], "frame": d["frame"], "phone_present": d["present"],
               "pte_present": float(present_pte.ravel()[0]), "torch_present": float(present_t.ravel()[0]),
               "phone_mask_frac": d["mask_frac"], "mask_frac": mask_frac, "brightness": d["brightness"],
               "mediapipe_hand": bool(hands), "landmarks_on_mask": lm_on_mask, **geo}
        rows.append(row)
        if len(tiles) < 60:
            caption = [f"#{d['dump']} phone {d['present']:.2f} pte {row['pte_present']:.2f}",
                       f"MP {'HAND' if hands else 'none'} size {geo['hand_size']:.2f}{' CUT' if geo['cut_off'] else ''}",
                       f"mask {mask_frac:.3f} lm-on-mask {lm_on_mask:.2f}"]
            tiles.append(tile(rgb, mask, hands, caption, bool(hands)))
    detector.close()
    if not rows:
        print("no dumps to analyse")
        return

    with (s / "diagnosis.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % COLS)
    cv2.imwrite(str(s / "diagnosis.jpg"), np.vstack([np.hstack(tiles[k:k + COLS]) for k in range(0, len(tiles), COLS)]))

    def col(k):
        return np.array([r[k] for r in rows], dtype=float)

    print(f"\nparity: median |phone - pte| present = {np.median(np.abs(col('phone_present') - col('pte_present'))):.4f}, "
          f"|pte - torch| = {np.median(np.abs(col('pte_present') - col('torch_present'))):.4f}")
    real = col("mediapipe_hand") > 0
    print(f"MediaPipe sees a hand in {real.sum()}/{len(rows)} NO-HAND dumps (= confirmed misses)")
    if real.any():
        found = col("mask_frac")[real] >= MASK_FOUND_FRACTION
        on = col("landmarks_on_mask")[real]
        print(f"  mask head still found the hand in {found.sum()}/{real.sum()} "
              f"(median landmarks on mask {np.nanmedian(on):.2f}) -> hand-present head is the one saying no")
        print(f"  both heads missed it in {(~found).sum()}/{real.sum()}")
        sizes = col("hand_size")[real]
        print(f"  hand size in crop: median {np.nanmedian(sizes):.2f} (10-90%: {np.nanpercentile(sizes, 10):.2f}-"
              f"{np.nanpercentile(sizes, 90):.2f}); cut off by the box in {col('cut_off')[real].sum():.0f}/{real.sum()}")
        print(f"  brightness: misses {np.median(col('brightness')[real]):.0f} vs all frames "
              f"{np.median([float(f['brightness']) for f in frames]):.0f}")
    if args.dataset.exists():
        detector = HandLandmarks()
        train_sizes = training_hand_sizes(args.dataset, detector)
        detector.close()
        print(f"training crops: hand size median {np.median(train_sizes):.2f} "
              f"(10-90%: {np.percentile(train_sizes, 10):.2f}-{np.percentile(train_sizes, 90):.2f})")
    print(f"\nwrote {s / 'diagnosis.jpg'} and {s / 'diagnosis.csv'}")


if __name__ == "__main__":
    main()
