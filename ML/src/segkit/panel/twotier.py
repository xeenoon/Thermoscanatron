"""Two-model tracking, simulated on a labelled video: a small model on every frame, a big one in the background.

    segkit-panel-twotier data/panel/<session> --small runs/panel_s1/panelseg_small.pte \\
        --big runs/panel_v3/panelseg.pte --big-every 6 --big-latency 3

fast path  every frame: the small model (low resolution) + PanelTracker follow/correct. Cheap enough for the
           camera's full frame rate on the phone.
slow path  every --big-every frames the big model runs on that frame; its answer is ready --big-latency frames
           later (it runs on its own thread on the phone). A tracker of its own, seeded with what the fast path
           believed at that frame, fits the big model's output (and renumbers by any visible edge, or acquires
           from scratch if the fast path had lost the panel). The result reorients the fast path: carried to the
           present through the fast path's own motion since that frame, H_now = H_fast(now) H_fast(t)^-1 H_big(t).

Prints cell-under-the-crosshair accuracy against the labels for: small only, big on every frame, big only at
the slow rate (coasting in between), and the two together.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from segkit.datasets.panels import is_val, load_session
from segkit.panel.track import Model, PanelTracker, cell_at, centre_crop

SIZE = 384


class TwoTier:
    def __init__(self, spec, big_every: int, big_latency: int):
        self.fast = PanelTracker(spec, SIZE, klt=False, block_match=True)
        self.slow = PanelTracker(spec, SIZE, klt=False, block_match=False)
        self.every, self.latency = big_every, big_latency
        self.history: dict[int, np.ndarray | None] = {}   # frame -> fast H
        self.pending: list[tuple[int, np.ndarray | None]] = []   # (frame the big model saw, its H)
        self.n = 0

    def step(self, crop_rgb, dense_small, present_small, big_fn) -> np.ndarray | None:
        i = self.n
        self.n += 1
        self.fast.step(crop_rgb, dense_small, present_small)
        # Big model: sees frame i now, answer lands at i + latency.
        if i % self.every == 0:
            dense_big, present_big = big_fn()
            seed = self.fast.H
            self.slow.H = None if seed is None else seed.copy()
            self.slow.misses = 0
            self.slow.prev_gray = None
            res = self.slow.step(crop_rgb, dense_big, present_big)
            good = res.state in ("tracking", "relocked", "acquired")
            self.pending.append((i, res.H if good else None))
        self.history[i] = None if self.fast.H is None else self.fast.H.copy()
        while self.pending and self.pending[0][0] + self.latency <= i:
            t, Hb = self.pending.pop(0)
            if Hb is None:
                continue
            Ht = self.history.get(t)
            if Ht is not None and self.fast.H is not None:
                self.fast.H = self.fast.H @ np.linalg.inv(Ht) @ Hb
            else:
                self.fast.H = Hb          # fast path had lost it: take the big model's (slightly stale) fix
            self.fast.misses = 0
        self.history.pop(i - 4 * self.every - self.latency, None)
        return self.fast.H


def main() -> None:
    ap = argparse.ArgumentParser(prog="segkit-panel-twotier", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("--small", type=Path, required=True)
    ap.add_argument("--small-size", type=int, default=192)
    ap.add_argument("--big", type=Path, required=True)
    ap.add_argument("--big-every", type=int, default=6)
    ap.add_argument("--big-latency", type=int, default=3)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    frames, Hs, valid, spec = load_session(args.session)
    small = Model(args.small, args.small_size)
    big = Model(args.big, SIZE)
    end = args.end or len(frames)
    runs = {
        "small only": PanelTracker(spec, SIZE, klt=False, block_match=True),
        "big every frame": PanelTracker(spec, SIZE, klt=False, block_match=True),
        f"big only, every {args.big_every}th frame": PanelTracker(spec, SIZE, klt=False, block_match=True),
    }
    two = TwoTier(spec, args.big_every, args.big_latency)
    results = {k: [] for k in [*runs, "two models"]}
    for i in tqdm(range(end)):
        rgb = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)
        crop, A = centre_crop(rgb, SIZE)
        small_in = cv2.resize(crop, (args.small_size, args.small_size), interpolation=cv2.INTER_AREA)
        ds, ps = small(small_in)
        ds = np.stack([cv2.resize(c, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR) for c in ds])
        cache = {}

        def big_fn():
            if "b" not in cache:
                cache["b"] = big(crop)
            return cache["b"]

        label = cell_at(A @ Hs[i], spec, SIZE / 2, SIZE / 2) if valid[i] else None
        out = {}
        out["small only"] = runs["small only"].step(crop, ds, ps).H
        db, pb = big_fn()
        out["big every frame"] = runs["big every frame"].step(crop, db, pb).H
        tr = runs[f"big only, every {args.big_every}th frame"]
        if i % args.big_every == 0:
            out[f"big only, every {args.big_every}th frame"] = tr.step(crop, db, pb).H
        else:
            # No model this frame: coast on image motion.
            out[f"big only, every {args.big_every}th frame"] = tr.coast(crop)
        out["two models"] = two.step(crop, ds, ps, big_fn)
        for k, H in out.items():
            cell = None if H is None else cell_at(H, spec, SIZE / 2, SIZE / 2)
            results[k].append((i, cell, label))

    for k, rs in results.items():
        for name, sel in (("all", rs), ("val blocks", [r for r in rs if is_val(r[0])])):
            on = [r for r in sel if r[2] is not None]
            said = [r for r in on if r[1] is not None]
            right = sum(r[1] == r[2] for r in said)
            print(f"{k:32s} {name:10s}: cell given {len(said) / max(1, len(on)):5.0%}, "
                  f"correct {right / max(1, len(said)):6.1%}  ({right}/{len(said)})")


if __name__ == "__main__":
    main()
