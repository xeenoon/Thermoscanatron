"""Hand segmentation samples from a segkit-label dataset dir (images/, masks/, index.csv, exclude.txt).

Each sample is a square crop around the hand mask, resized to `size` — a stand-in for the
MediaPipe ROI crop the phone pipeline will use. Training crops jitter scale, centre and rotation.

Negatives ("no hand here"): frames labelled no_hand (empty mask) give random crops, and a share of
training samples from hand frames are background crops chosen to miss the hand. Every sample also
returns a hand-present target: 1 if at least PRESENT_MIN_FRACTION of the crop is hand.
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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FRAME_RE = re.compile(r"_f(\d+)$")
VAL_BLOCK_FRAMES = 90   # 3 s of 30 fps video per block
VAL_EVERY = 5           # every 5th block is validation: ~20%, and no near-duplicate frames leak across
SKIP_FLAGS = {"landmarks_outside"}  # rembg mask disagrees with MediaPipe: not trusted
BACKGROUND_CROP_P = 0.25            # share of hand-frame training samples cropped away from the hand
PRESENT_MIN_FRACTION = 0.005        # crop counts as "hand present" above this hand-pixel fraction


def normalize(rgb_u8: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> contiguous CHW float tensor, ImageNet-normalised."""
    x = (rgb_u8.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


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


class HandCrops(Dataset):
    """train: random hand/background crops. Otherwise deterministic: a centred hand crop, or with
    background_only, a seeded crop that misses the hand (validation negatives)."""

    def __init__(self, dataset: Path, stems: list[str], size: int = 384, train: bool = True,
                 background_only: bool = False):
        self.dataset, self.stems, self.size, self.train = dataset, stems, size, train
        self.background_only = background_only
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

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stem = self.stems[i]
        rgb = cv2.cvtColor(cv2.imread(str(self.dataset / "images" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.dataset / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        rng = np.random.default_rng(i if self.background_only else None)

        m = None
        if self.background_only and mask.any():
            m = self.background_crop(mask, rng)
        if m is None and not mask.any():
            # No-hand frame: any crop of it is a negative.
            if self.train:
                side = rng.uniform(0.3, 0.9) * min(h, w)
                x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
                m = crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.uniform(-180, 180))
            else:
                side = 0.6 * min(h, w)
                m = crop_affine(((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), self.size, 1.0,
                                (0, 0), 0.0)
        elif self.train and rng.random() < BACKGROUND_CROP_P:
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
        msk = cv2.warpAffine(mask, m, (self.size, self.size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 128

        if self.train:
            if rng.random() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            img = self.photometric(image=np.ascontiguousarray(img))["image"]
        present = torch.tensor([float(msk.mean() >= PRESENT_MIN_FRACTION)])
        return normalize(img), torch.from_numpy(np.ascontiguousarray(msk, dtype=np.float32))[None], present
