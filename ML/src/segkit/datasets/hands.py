"""Hand segmentation samples from a segkit-label dataset dir (images/, masks/, index.csv, exclude.txt).

Each sample is a square crop around the hand mask, resized to `size` — a stand-in for the
MediaPipe ROI crop the phone pipeline will use. Training crops jitter scale, centre and rotation.

Negatives ("no hand here"): frames labelled no_hand (empty mask) give random crops, and a share of
training samples from hand frames are background crops chosen to miss the hand. Every sample also
returns a hand-present target: 1 if at least PRESENT_MIN_FRACTION of the crop is hand.

Background swap (training only): a share of hand crops get everything outside the hand mask replaced by
a hand-free crop from a random other frame, so the model cannot learn "hand = whatever isn't this room".
"""

import csv
import math
import re
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segkit.skin_tone import random_tone, recolour

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FRAME_RE = re.compile(r"_f(\d+)$")
VAL_BLOCK_FRAMES = 90   # 3 s of 30 fps video per block
VAL_EVERY = 5           # every 5th block is validation: ~20%, and no near-duplicate frames leak across
SKIP_FLAGS = {"landmarks_outside", "person_no_skin"}  # rembg mask disagrees with MediaPipe: not trusted
BACKGROUND_CROP_P = 0.25            # share of hand-frame training samples cropped away from the hand
PRESENT_MIN_FRACTION = 0.005        # crop counts as "hand present" above this hand-pixel fraction
BACKGROUND_SWAP_P = 0.5             # share of hand crops whose background is replaced (training only)
EXTERNAL_BACKGROUND_P = 0.7         # of those, share whose new background comes from the negatives list
SWAP_FEATHER_PX = 2                 # soften the pasted edge so the model can't key on a hard seam


def normalize(rgb_u8: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> contiguous CHW float tensor, ImageNet-normalised."""
    x = (rgb_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def session_of(stem: str) -> str:
    """Recording session (video) a frame came from: the stem minus its _f<frame> suffix."""
    return FRAME_RE.sub("", stem)


def is_val(stem: str) -> bool:
    m = FRAME_RE.search(stem)
    return m is not None and (int(m.group(1)) // VAL_BLOCK_FRAMES) % VAL_EVERY == VAL_EVERY // 2


def load_split(dataset: Path) -> tuple[list[str], list[str]]:
    rows = list(csv.DictReader((dataset / "index.csv").open()))
    stems = [r["stem"] for r in rows if not SKIP_FLAGS & set(r["flags"].split("|"))]
    exclude_path = dataset / "exclude.txt"
    excluded = set()
    if exclude_path.exists():
        excluded = {line.split("#")[0].strip() for line in exclude_path.read_text().splitlines()} - {""}
    stems = [s for s in stems if s not in excluded]
    return [s for s in stems if not is_val(s)], [s for s in stems if is_val(s)]


def crop_affine(bbox: tuple[int, int, int, int], size: int, scale: float, shift: tuple[float, float],
                angle_deg: float) -> np.ndarray:
    """2x3 affine mapping a rotated square around the bbox onto a size x size output."""
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) * scale
    cx = (x0 + x1) / 2 + shift[0] * side
    cy = (y0 + y1) / 2 + shift[1] * side
    s = size / side
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) * s, math.sin(a) * s
    # out = R*s*(p - c) + size/2
    return np.array([[cos, -sin, size / 2 - cos * cx + sin * cy],
                     [sin, cos, size / 2 - sin * cx - cos * cy]], np.float32)


def random_square(rgb: np.ndarray, size: int, rng: np.random.Generator | None, train: bool = True) -> np.ndarray:
    """Random (train) or centred square crop of any image, resized to size x size."""
    h, w = rgb.shape[:2]
    if train:
        side = rng.uniform(0.4, 1.0) * min(h, w)
        x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
        m = crop_affine((x0, y0, x0 + side, y0 + side), size, 1.0, (0, 0), rng.uniform(-180, 180))
    else:
        side = min(h, w)
        m = crop_affine(((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), size, 1.0, (0, 0), 0.0)
    return cv2.warpAffine(rgb, m, (size, size), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REFLECT_101)


class NegativeImages(Dataset):
    """Hand-free photos (e.g. COCO without people): empty mask, hand-present = 0."""

    def __init__(self, paths: list[Path], size: int = 384, train: bool = True, extras: bool = False,
                 prev_mask: bool = False):
        self.paths, self.size, self.train = paths, size, train
        self.extras, self.prev_mask = extras, prev_mask
        self.photometric = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.OneOf([A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=(3, 5))], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = cv2.cvtColor(cv2.imread(str(self.paths[i])), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(None if self.train else i)
        img = random_square(rgb, self.size, rng, self.train)
        if self.train:
            img = self.photometric(image=img)["image"]
        x = normalize(img)
        if self.prev_mask:
            empty = np.zeros((self.size, self.size), bool)
            x = torch.cat([x, torch.from_numpy(prev_mask_channel(empty, rng, self.train))[None]])
        return x, torch.zeros(3 if self.extras else 1, self.size, self.size), torch.zeros(1)


AUX_CLASSES = 4                     # parse/ maps: 0 background, 1 skin, 2 hair, 3 clothing


def prev_mask_channel(msk: np.ndarray, rng: np.random.Generator, train: bool) -> np.ndarray:
    """A stand-in for the previous frame's mask, for the per-frame model's 4th input channel.

    Training: the true mask moved / scaled / rotated a little (motion), grown or shrunk (the last prediction's
    errors), sometimes empty (first frame, lost track) or replaced by a random blob (a stale wrong answer), so the
    model learns to use it as a hint, never to copy it. Validation: the true mask shifted by a fixed 6 px."""
    n = msk.shape[0]
    m = msk.astype(np.uint8)
    if not train:
        return np.roll(m, 6, axis=1).astype(np.float32)
    r = rng.random()
    if r < 0.15 or not m.any():
        out = np.zeros_like(m)
        if rng.random() < 0.3:   # a wrong blob somewhere
            c = rng.uniform(0.2, 0.8, 2) * n
            cv2.ellipse(out, (int(c[0]), int(c[1])), (int(rng.uniform(10, n / 4)), int(rng.uniform(10, n / 4))),
                        rng.uniform(0, 180), 0, 360, 1, -1)
        return out.astype(np.float32)
    a = math.radians(rng.normal(0, 6))
    sc = rng.uniform(0.9, 1.1)
    t = rng.normal(0, 0.05 * n, 2)
    M = np.array([[sc * math.cos(a), -sc * math.sin(a), 0], [sc * math.sin(a), sc * math.cos(a), 0]], np.float32)
    M[:, 2] = (n / 2) - M[:, :2] @ np.array([n / 2, n / 2]) + t
    out = cv2.warpAffine(m, M, (n, n), flags=cv2.INTER_NEAREST)
    k = int(rng.integers(1, 9)) | 1
    out = (cv2.dilate if rng.random() < 0.5 else cv2.erode)(out, np.ones((k, k), np.uint8))
    return out.astype(np.float32)


def read_list(path: Path | None) -> list[Path]:
    if path is None:
        return []
    return [Path(line) for line in path.read_text().splitlines() if line.strip()]


class HandCrops(Dataset):
    """train: random hand/background crops. Otherwise deterministic: a centred hand crop, or with
    background_only, a seeded crop that misses the hand (validation negatives)."""

    def __init__(self, dataset: Path, stems: list[str], size: int = 384, train: bool = True,
                 background_only: bool = False, negatives: list[Path] | None = None, skin_tone_p: float = 0.0,
                 frame_crop_p: float = 0.0, phone_view: bool = False, extras: bool = False,
                 prev_mask: bool = False):
        self.dataset, self.stems, self.size, self.train = dataset, stems, size, train
        self.background_only = background_only
        self.skin_tone_p = skin_tone_p   # share of training crops whose labelled skin gets a random tone
        # Share of training crops taken like the phone's view: a random square of the whole frame (any position,
        # 45-100% of the short side, near-upright), so skin can be small, off-centre, several, or run off the edge.
        self.frame_crop_p = frame_crop_p
        self.phone_view = phone_view     # validation: the phone's crop (centred square, 90% of the short side)
        self.extras = extras             # target = [skin, hand, parse class] instead of [skin] (hands/, parse/ dirs)
        self.prev_mask = prev_mask       # input gets a 4th channel: a perturbed copy of the mask (prev_mask_channel)
        self.negatives = negatives or []
        self.photometric = A.Compose([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.8),
            A.OneOf([A.MotionBlur(blur_limit=7), A.GaussianBlur(blur_limit=(3, 5))], p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ])

    def __len__(self) -> int:
        return len(self.stems)

    def background_crop(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        """Affine for a random square that (almost) misses the hand, or None if none found."""
        h, w = mask.shape
        for _ in range(10):
            side = rng.uniform(0.2, 0.6) * min(h, w)
            x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
            if (mask[int(y0):int(y0 + side), int(x0):int(x0 + side)] > 0).mean() < PRESENT_MIN_FRACTION:
                return crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.uniform(-180, 180))
        return None

    def load(self, stem: str) -> tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(cv2.imread(str(self.dataset / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        return rgb, cv2.imread(str(self.dataset / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)

    def random_background(self, rng: np.random.Generator) -> np.ndarray | None:
        """A size x size hand-free crop: from the negatives list, or from a frame of our own (away from the hand)."""
        if self.negatives and rng.random() < EXTERNAL_BACKGROUND_P:
            bgr = cv2.imread(str(self.negatives[rng.integers(len(self.negatives))]))
            if bgr is not None:
                return random_square(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.size, rng)
        for _ in range(3):
            rgb, mask = self.load(self.stems[rng.integers(len(self.stems))])
            h, w = mask.shape
            if mask.any():
                m = self.background_crop(mask, rng)
            else:
                side = rng.uniform(0.3, 0.9) * min(h, w)
                x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
                m = crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.uniform(-180, 180))
            if m is not None:
                return cv2.warpAffine(rgb, m, (self.size, self.size), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT_101)
        return None

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[i]
        rgb, mask = self.load(stem)
        h, w = mask.shape
        rng = np.random.default_rng(i if self.background_only else None)

        m = None
        frame_crop = False
        if not self.train and self.phone_view and not self.background_only:
            side = 0.9 * min(h, w)
            m = crop_affine(((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), self.size, 1.0,
                            (0, 0), 0.0)
        elif self.train and mask.any() and rng.random() < self.frame_crop_p:
            side = rng.uniform(0.45, 1.0) * min(h, w)
            x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
            m = crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.normal(0, 15))
            frame_crop = True
        if m is None and self.background_only and mask.any():
            m = self.background_crop(mask, rng)
        if m is None and not mask.any() and not (not self.train and self.phone_view):
            # No-hand frame: any crop of it is a negative.
            if self.train:
                side = rng.uniform(0.3, 0.9) * min(h, w)
                x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
                m = crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.uniform(-180, 180))
            else:
                side = 0.6 * min(h, w)
                m = crop_affine(((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), self.size, 1.0,
                                (0, 0), 0.0)
        elif m is None and self.train and rng.random() < BACKGROUND_CROP_P:
            m = self.background_crop(mask, rng)
        if m is None:
            ys, xs = np.nonzero(mask)
            bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            if self.train:
                m = crop_affine(bbox, self.size, rng.uniform(1.1, 1.6), tuple(rng.uniform(-0.15, 0.15, 2)),
                                rng.uniform(-180, 180))
            else:
                m = crop_affine(bbox, self.size, 1.3, (0.0, 0.0), 0.0)

        img = cv2.warpAffine(rgb, m, (self.size, self.size), flags=cv2.INTER_AREA if not self.train else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        extra = None
        if self.extras:
            hand = cv2.imread(str(self.dataset / "hands" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            parse = cv2.imread(str(self.dataset / "parse" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
            hand = np.zeros_like(mask) if hand is None else hand
            parse = (mask > 0).astype(np.uint8) if parse is None else parse
            extra = [cv2.warpAffine(hand, m, (self.size, self.size), flags=cv2.INTER_LINEAR) >= 128,
                     cv2.warpAffine(parse, m, (self.size, self.size), flags=cv2.INTER_NEAREST)]
        msk = cv2.warpAffine(mask, m, (self.size, self.size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 128

        if self.train and not frame_crop and msk.any() and rng.random() < BACKGROUND_SWAP_P:
            bg = self.random_background(rng)
            if bg is not None:
                # Hand stays, everything else (incl. the black out-of-frame padding) becomes the new scene.
                k = 2 * SWAP_FEATHER_PX + 1
                alpha = cv2.GaussianBlur(msk.astype(np.float32), (k, k), 0)[..., None]
                img = (alpha * img + (1 - alpha) * bg).astype(np.uint8)
                if extra is not None:
                    extra[1] = np.zeros_like(extra[1])
        if self.train:
            if self.skin_tone_p and msk.any() and rng.random() < self.skin_tone_p:
                img = recolour(np.ascontiguousarray(img), msk, *random_tone(rng))
            if rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
                if extra is not None:
                    extra = [e[:, ::-1] for e in extra]
            img = self.photometric(image=np.ascontiguousarray(img))["image"]
        present = torch.tensor([float(msk.mean() >= PRESENT_MIN_FRACTION)])
        x = normalize(img)
        if self.prev_mask:
            x = torch.cat([x, torch.from_numpy(prev_mask_channel(np.ascontiguousarray(msk), rng, self.train))[None]])
        target = [np.ascontiguousarray(msk, dtype=np.float32)]
        if extra is not None:
            # Background swap replaced everything off the skin: the parse classes there are background now.
            parse = np.where(msk, 1, np.where(extra[1] == 1, 0, extra[1])).astype(np.float32)
            target += [np.ascontiguousarray(extra[0] & msk, dtype=np.float32), np.ascontiguousarray(parse)]
        return x, torch.from_numpy(np.stack(target)), present
