"""Solar panel samples from segkit-panel-label sessions (frames/ + panel_labels.npz).

Each sample is a square crop of a video frame (random size, position and rotation in training; a centred
crop like the phone's analysis box otherwise) with dense targets computed from the frame's panel
homography, mapped through the crop:

  mask   [1]  pixel lies in the cell area
  lines  [1]  pixel lies on a gridline (cell gap), width from the panel profile, at least LINE_MIN_PX
  phase  [4]  sin/cos(2 pi u), sin/cos(pi v): where in its cell the pixel is. u repeats every cell; v
              every two rows, because the diamonds only mark every second row boundary, so the row
              parity is visible locally. Only meaningful inside the mask.
  present     share of the crop covered by the cell area >= PRESENT_MIN_FRACTION

The integer cell index is deliberately not a target: a close-up of one cell looks like any other.
That part is the tracker's job (segkit.panel.track).

Negatives: frames from other recordings that contain no panel (e.g. the hand videos).
"""

import json
import math
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from segkit.datasets.hands import crop_affine, normalize, random_square
from segkit.panel.spec import PanelSpec

VAL_BLOCK_FRAMES = 90   # 3 s of 30 fps video per block
VAL_EVERY = 5           # every 5th block is validation
PRESENT_MIN_FRACTION = 0.02
LINE_MIN_PX = 1.5
N_TARGETS = 6           # mask, lines, 4 x phase
MIN_CROP = 0.12         # smallest training crop, as a fraction of the frame's short side
PERSPECTIVE_P = 0.5     # share of training crops given an extra random perspective tilt
PERSPECTIVE_JITTER = 0.15  # corner displacement, as a fraction of the crop side
ROT90_P = 0.25          # with rot90: share of training crops turned a quarter, half or three-quarter turn


def load_session(session: Path) -> tuple[list[Path], np.ndarray, np.ndarray, PanelSpec]:
    d = np.load(session / "panel_labels.npz")
    spec_d = json.loads(str(d["spec"]))
    spec = PanelSpec(**{**spec_d, "diamond_rows": tuple(spec_d["diamond_rows"])})
    frames = sorted((session / "frames").glob("*.jpg"))
    return frames, d["H"], d["valid"], spec


def session_is_crops(session: Path) -> bool:
    with np.load(session / "panel_labels.npz") as d:
        return str(d.get("coordinate_space", "video")) == "analysis_crop"


def grazing_transform(size: int, rng: np.random.Generator) -> np.ndarray:
    """Project a plane tilted up to 72 degrees; retain positive depth and a convex quadrilateral."""
    yaw, pitch = np.deg2rad(rng.uniform(-72, 72)), np.deg2rad(rng.uniform(-35, 35))
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rotation = np.array([[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]])
    K = np.array([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1.]])
    plane = rotation.copy()
    plane[:, 2] = [0, 0, size]
    return K @ plane @ np.array([[1, 0, -size / 2], [0, 1, -size / 2], [0, 0, 1.]])


def is_val(frame_index: int) -> bool:
    return (frame_index // VAL_BLOCK_FRAMES) % VAL_EVERY == VAL_EVERY // 2


def targets(H_crop: np.ndarray, size: int, spec: PanelSpec) -> np.ndarray:
    """[N_TARGETS, size, size] float32 targets for a crop whose panel->crop homography is H_crop."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    uvw = np.linalg.inv(H_crop) @ pts
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (uvw[0] / uvw[2]).reshape(size, size)
        v = (uvw[1] / uvw[2]).reshape(size, size)
    front = (uvw[2] > 0).reshape(size, size) if np.linalg.det(H_crop) > 0 else (uvw[2] < 0).reshape(size, size)
    inside = front & (u >= 0) & (u <= spec.cols) & (v >= 0) & (v <= spec.rows)

    # Gridline half-width in panel units: the profile's gap width, but at least LINE_MIN_PX in the crop
    # (np.gradient gives panel units per pixel).
    half_u = np.maximum(spec.line_frac / 2, LINE_MIN_PX / 2 * np.hypot(*np.gradient(u)))
    half_v = np.maximum(spec.line_frac / 2, LINE_MIN_PX / 2 * np.hypot(*np.gradient(v)))
    lines = inside & ((np.abs(u - np.round(u)) <= half_u) | (np.abs(v - np.round(v)) <= half_v))

    out = np.zeros((N_TARGETS, size, size), np.float32)
    out[0] = inside
    out[1] = lines
    out[2] = np.sin(2 * math.pi * u) * inside
    out[3] = np.cos(2 * math.pi * u) * inside
    out[4] = np.sin(math.pi * v) * inside
    out[5] = np.cos(math.pi * v) * inside
    return np.nan_to_num(out)


def affine3(m: np.ndarray) -> np.ndarray:
    return np.vstack([m, [0, 0, 1]]).astype(np.float64)


class PanelCrops(Dataset):
    def __init__(self, sessions: list[Path], split: str, size: int = 320, train: bool = True,
                 negatives: list[Path] | None = None, grazing: bool = False, rot90: bool = False):
        self.size, self.train = size, train
        self.grazing = grazing
        # Phone held sideways or a panel lying on its side: the 20-degree crop jitter never gets there.
        self.rot90 = rot90
        self.crop_paths = set()
        self.items: list[tuple[Path, np.ndarray | None]] = []
        self.spec = None
        for s in sessions:
            frames, Hs, valid, spec = load_session(s)
            self.spec = self.spec or spec
            if session_is_crops(s):
                self.crop_paths.update(frames)
            for i, f in enumerate(frames):
                if valid[i] and is_val(i) == (split == "val"):
                    self.items.append((f, Hs[i]))
        self.n_panel = len(self.items)
        self.items += [(p, None) for p in (negatives or [])]
        # The training video is one overcast afternoon; the panel also gets used indoors, at night, with the
        # operator's shadow and reflection across the glass. So: dark exposures, shadows, glare, sensor noise.
        self.photometric = A.Compose([
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05, p=0.8),
            A.OneOf([
                A.RandomGamma(gamma_limit=(100, 300)),
                A.RandomBrightnessContrast(brightness_limit=(-0.75, 0.1), contrast_limit=(-0.5, 0.1)),
            ], p=0.5),
            A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=6, shadow_roi=(0, 0, 1, 1),
                           shadow_intensity_range=(0.3, 0.8), p=0.4),
            A.RandomSunFlare(src_radius=120, p=0.1),
            A.OneOf([A.MotionBlur(blur_limit=9), A.GaussianBlur(blur_limit=(3, 7))], p=0.4),
            A.OneOf([A.GaussNoise(std_range=(0.01, 0.06)), A.ISONoise(intensity=(0.1, 0.6))], p=0.5),
            A.ImageCompression(quality_range=(50, 95), p=0.3),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def crop(self, w: int, h: int, rng: np.random.Generator) -> np.ndarray:
        if self.train:
            # From the phone's whole analysis box down to a 12% zoom (log-uniform, so extreme close-ups, where
            # one cell or less fills the view, get as much training as far views), anywhere in the frame.
            side = math.exp(rng.uniform(math.log(MIN_CROP), 0.0)) * min(h, w)
            x0, y0 = rng.uniform(0, w - side), rng.uniform(0, h - side)
            return crop_affine((x0, y0, x0 + side, y0 + side), self.size, 1.0, (0, 0), rng.normal(0, 20))
        # The phone's box: centred square, 90% of the short side.
        side = 0.9 * min(h, w)
        return crop_affine(((w - side) / 2, (h - side) / 2, (w + side) / 2, (h + side) / 2), self.size, 1.0,
                           (0, 0), 0.0)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path, H = self.items[i]
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        rng = np.random.default_rng(None if self.train else i)
        if H is None:
            img = random_square(rgb, self.size, rng, self.train)
            t = np.zeros((N_TARGETS, self.size, self.size), np.float32)
        else:
            h, w = rgb.shape[:2]
            if not self.train and path in self.crop_paths:
                M = np.diag([self.size / w, self.size / h, 1.])
            elif self.train and path in self.crop_paths and rng.random() < 0.5:
                M = np.diag([self.size / w, self.size / h, 1.])
            else:
                M = affine3(self.crop(w, h, rng))
            if self.train and self.rot90 and rng.random() < ROT90_P:
                c = self.size / 2
                R = cv2.getRotationMatrix2D((c, c), 90.0 * rng.integers(1, 4), 1.0)
                M = affine3(R) @ M
            if self.train and rng.random() < PERSPECTIVE_P:
                # Random off-angle view: move the crop's corners independently (a homography on top of the crop).
                n = self.size
                src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
                dst = src + rng.uniform(-PERSPECTIVE_JITTER, PERSPECTIVE_JITTER, (4, 2)).astype(np.float32) * n
                tilt = (grazing_transform(n, rng) if self.grazing and rng.random() < 0.5 else
                        cv2.getPerspectiveTransform(src, dst).astype(np.float64))
                M = tilt @ M
            img = cv2.warpPerspective(rgb, M, (self.size, self.size), flags=cv2.INTER_AREA,
                                      borderMode=cv2.BORDER_CONSTANT)
            t = targets(M @ H, self.size, self.spec)
            support = cv2.warpPerspective(np.ones((h, w), np.uint8), M, (self.size, self.size),
                                          flags=cv2.INTER_NEAREST)
            t *= support[None]
        if self.train:
            img = self.photometric(image=img)["image"]
        present = torch.tensor([float(t[0].mean() >= PRESENT_MIN_FRACTION)])
        return normalize(img), torch.from_numpy(t), present
