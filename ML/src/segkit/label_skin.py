"""Auto-label exposed human skin (faces, hands, arms, legs) with a human-parsing model.

    segkit-label-skin data/videos/clip.mp4 [more.mp4 | frame_dir ...] --out data/skin_v1 --fps 5

The target is every visible patch of skin of any person, as separate blobs if need be, and nothing else:
not clothing (sleeves, hoodies, gloves), hair, plants, animals or paintings.

Labeller: SegFormer-B2 trained on ATR human parsing (mattmdjaga/segformer_b2_clothes, 18 classes). It labels a
person's parts from shape and context (where a face, an arm, a leg is), not from colour, so it does not key on
skin tone the way colour-threshold skin detectors do. Skin = Face + Left/Right-arm + Left/Right-leg (the arm
classes include the hands). Hair, hats, sunglasses and every clothing class are left out.

Close-up hands: with no body in view the parser only half-labels a hand that fills the frame (fingers yes, palm
patchy). So where MediaPipe finds a hand, the old hand labeller's outline is added: rembg's blob touching the
landmarks, cut at the wrist and boxed to the hand (segkit.label_rembg.restrict_to_hands) - which keeps sleeves
out. Skin mask = parser skin OR hand outlines.

Clean-up: pixels need SKIN_PROB of the parsing model's probability on the skin classes; blobs smaller than
MIN_BLOB of the frame are dropped (fur, wood grain, skin-coloured paint), small holes are filled.

Writes the hand-dataset layout (images/, masks/, overlays/, index.csv with flags), so segkit-validate,
segkit-audit, segkit-train and the review/exclude.txt workflow all work unchanged. Flags: no_skin (empty mask:
a negative), tiny (largest blob < 0.5% of the frame), dropped (clean-up removed > 20% of the raw skin), blurry.
Frame stems are <video>_f<frame>, which also gives the validation split its time blocks.
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PARSER = "mattmdjaga/segformer_b2_clothes"
SKIN_CLASSES = (11, 12, 13, 14, 15)   # Face, Left-leg, Right-leg, Left-arm, Right-arm
SKIN_PROB = 0.5
MIN_BLOB = 0.0008                     # of the frame
MIN_BLOB_CONF = 0.8                   # a parser-only blob also needs this mean skin probability (fur, wood)
FILL_HOLES_BELOW = 0.002              # holes smaller than this (of the frame) inside a skin blob are filled
TINY = 0.005
DROPPED_FLAG = 0.2
MIN_EDGE_SHARPNESS = 20.0
OVERLAY_SIDE = 640
FIELDS = ["stem", "source", "width", "height", "hands", "area", "discarded", "lm_in_mask", "edge_sharpness",
          "flags", "model"]


class HandOutlines:
    """MediaPipe hand landmarks -> rembg blob(s) at those landmarks, wrist-cut (the old hand labeller)."""

    def __init__(self):
        import onnxruntime as ort
        from rembg import new_session
        from segkit.hand_landmarks import HandLandmarks
        self.detector = HandLandmarks()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = new_session("birefnet-general-lite", providers=providers)

    def __call__(self, rgb: np.ndarray) -> np.ndarray | None:
        from PIL import Image
        from rembg import remove
        from segkit.label_rembg import keep_components, restrict_to_hands
        lms = self.detector.detect(rgb)
        if not lms:
            return None
        alpha = np.asarray(remove(Image.fromarray(rgb), session=self.session, only_mask=True))
        mask, _ = keep_components(alpha, lms)
        mask, _ = restrict_to_hands(mask, lms)
        return mask


class SkinParser:
    def __init__(self, name: str = PARSER, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.net = AutoModelForSemanticSegmentation.from_pretrained(name).to(self.device).eval()

    @torch.no_grad()
    def skin_prob(self, rgb: np.ndarray) -> np.ndarray:
        """Per-pixel probability of skin (sum over the skin classes), full resolution."""
        inp = self.proc(images=np.ascontiguousarray(rgb), return_tensors="pt").to(self.device)
        logits = self.net(**inp).logits
        logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        p = logits.softmax(1)[0, list(SKIN_CLASSES)].sum(0)
        return p.float().cpu().numpy()


def clean(prob: np.ndarray, hands: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Threshold, add hand outlines, drop small / unsure blobs, fill small holes.
    Returns (0/255 mask, share of raw skin dropped)."""
    raw = (prob >= SKIN_PROB).astype(np.uint8)
    if hands is not None:
        raw |= (hands > 0).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    keep = np.zeros(n, bool)
    if n > 1:
        conf = np.bincount(cc.ravel(), weights=prob.ravel(), minlength=n) / np.maximum(st[:, cv2.CC_STAT_AREA], 1)
        from_hand = np.zeros(n, bool)
        if hands is not None:
            from_hand[np.unique(cc[hands > 0])] = True
        keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= MIN_BLOB * raw.size) & ((conf[1:] >= MIN_BLOB_CONF) | from_hand[1:])
    mask = keep[cc].astype(np.uint8)
    # Fill holes (glasses frames, specular highlights) that are small and fully enclosed.
    inv = 1 - mask
    n2, cc2, st2, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = mask.shape
    for k in range(1, n2):
        x, y, bw, bh, area = st2[k]
        if area < FILL_HOLES_BELOW * mask.size and x > 0 and y > 0 and x + bw < w and y + bh < h:
            mask[cc2 == k] = 1
    raw_area = int(raw.sum())
    dropped = 1 - int((mask & raw).sum()) / raw_area if raw_area else 0.0
    return mask * 255, dropped


def edge_sharpness(rgb: np.ndarray, mask: np.ndarray) -> float:
    k = np.ones((7, 7), np.uint8)
    band = cv2.dilate(mask, k) > cv2.erode(mask, k)
    if not band.any():
        return 0.0
    return float(cv2.Laplacian(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), cv2.CV_32F)[band].var())


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = OVERLAY_SIDE / max(rgb.shape[:2])
    vis = cv2.cvtColor(cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
    m = cv2.resize(mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis[m > 0] = (0.6 * vis[m > 0] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cs, -1, (0, 0, 255), 1)
    if not m.any():
        cv2.putText(vis, "NO SKIN", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return vis


def frames(source: Path, fps: float):
    """(stem, rgb) for a video (sampled at fps) or every *.jpg under a directory."""
    if source.is_dir():
        for p in sorted(source.rglob("*.jpg")):
            yield p.stem, cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(str(source))
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(native / fps))
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield f"{source.stem}_f{i:06d}", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-label-skin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", type=Path, nargs="+", help="videos and/or directories of jpgs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=5.0, help="frames per second taken from videos")
    args = ap.parse_args()
    for sub in ("images", "masks", "overlays"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    index = args.out / "index.csv"
    done = set()
    if index.exists():
        done = {r["stem"] for r in csv.DictReader(index.open())}
    parser = SkinParser()
    hand_outlines = HandOutlines()
    new = not index.exists()
    with index.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        n = 0
        t0 = time.perf_counter()
        for src in args.sources:
            for stem, rgb in frames(src, args.fps):
                if stem in done:
                    continue
                hands = hand_outlines(rgb)
                mask, dropped = clean(parser.skin_prob(rgb), hands)
                area = float((mask > 0).mean())
                flags = []
                if not mask.any():
                    flags.append("no_skin")
                else:
                    _, _, st, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
                    if st[1:, cv2.CC_STAT_AREA].max() < TINY * mask.size:
                        flags.append("tiny")
                    if dropped > DROPPED_FLAG:
                        flags.append("dropped")
                sharp = edge_sharpness(rgb, mask)
                if mask.any() and sharp < MIN_EDGE_SHARPNESS:
                    flags.append("blurry")
                cv2.imwrite(str(args.out / "images" / f"{stem}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(args.out / "masks" / f"{stem}.png"), mask)
                cv2.imwrite(str(args.out / "overlays" / f"{stem}.jpg"), overlay(rgb, mask))
                w.writerow({"stem": stem, "source": str(src), "width": rgb.shape[1], "height": rgb.shape[0],
                            "hands": "", "area": f"{area:.4f}", "discarded": f"{dropped:.4f}", "lm_in_mask": "",
                            "edge_sharpness": f"{sharp:.1f}", "flags": "|".join(flags), "model": PARSER})
                n += 1
                if n % 100 == 0:
                    f.flush()
                    print(f"{n} frames ({n / (time.perf_counter() - t0):.1f}/s), last {stem} "
                          f"{' '.join(flags) or 'ok'}", flush=True)
    print(f"done: {n} new frames in {args.out}")


if __name__ == "__main__":
    main()
