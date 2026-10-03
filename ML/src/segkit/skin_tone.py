"""Counterfactual skin-tone recolouring: the same frame, the same person, a different skin tone.

Used two ways:
  - training augmentation (segkit-train --skin-tone-aug P): a share of training crops get the labelled skin
    recoloured to a random tone, so the model cannot use "this particular colour" as the cue for skin;
  - fairness check (segkit-skin-eval): held-out frames are recoloured to each of TONES and scored again.
    A model that finds skin by shape and context should score about the same on every tone.

Recolouring works in CIELAB: inside the skin mask each pixel keeps its own variation around the skin's mean
(texture, shading, highlights) while the mean moves to the target tone; lightness variation is scaled with
the target lightness so dark tones are not washed out. The mask edge is feathered so no seam marks it.

Tones are approximate CIELAB means of skin across the Fitzpatrick scale (I lightest .. VI darkest), the
spread reported in skin-colorimetry studies of individual typology angle (ITA). They are a test grid, not a
claim about any person's appearance.
"""

import cv2
import numpy as np

# (name, L*, a*, b*)
TONES = [
    ("I", 74.0, 8.0, 14.0),
    ("II", 67.0, 11.0, 17.0),
    ("III", 59.0, 13.0, 20.0),
    ("IV", 49.0, 14.0, 22.0),
    ("V", 38.0, 13.0, 19.0),
    ("VI", 27.0, 10.0, 13.0),
]
FEATHER_PX = 3


def recolour(rgb: np.ndarray, mask: np.ndarray, L: float, a: float, b: float) -> np.ndarray:
    """rgb uint8 HxWx3, mask bool/0-255 HxW -> rgb with the masked skin moved to CIELAB mean (L, a, b)."""
    m = mask > 0
    if m.sum() < 20:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    # OpenCV 8-bit LAB: L in 0..255 (= L* x 2.55), a/b offset by 128.
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    mean = lab[m].mean(0)
    new = lab.copy()
    scale_l = L / max(mean[0], 5.0)
    new[..., 0] = L + (lab[..., 0] - mean[0]) * scale_l
    new[..., 1] = a + (lab[..., 1] - mean[1])
    new[..., 2] = b + (lab[..., 2] - mean[2])
    new[..., 0] = np.clip(new[..., 0], 0, 100) * 255 / 100
    new[..., 1:] = np.clip(new[..., 1:] + 128, 0, 255)
    out = cv2.cvtColor(new.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    k = 2 * FEATHER_PX + 1
    alpha = cv2.GaussianBlur(m.astype(np.float32), (k, k), 0)[..., None]
    return (alpha * out + (1 - alpha) * rgb).clip(0, 255).astype(np.uint8)


def random_tone(rng: np.random.Generator) -> tuple[float, float, float]:
    """A tone anywhere in the TONES range (interpolated), for augmentation."""
    t = rng.uniform(0, len(TONES) - 1)
    i = int(t)
    j = min(i + 1, len(TONES) - 1)
    f = t - i
    return tuple((1 - f) * np.array(TONES[i][1:]) + f * np.array(TONES[j][1:]))
