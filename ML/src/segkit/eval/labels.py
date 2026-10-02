"""Load LabelMe JSON (polygons) into full-resolution masks.

Labels:
  hand    outer outline of a hand, cut straight across at the wrist
  hole    background enclosed by a hand (e.g. thumb touching index finger)
  ignore  region excluded from boundary scoring (draw one across the wrist cut)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon


@dataclass
class Label:
    stem: str
    image_path: Path
    hand: np.ndarray    # bool [H, W]
    ignore: np.ndarray  # bool [H, W]


def rasterize(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Union of polygons in LabelMe coords (pixel i spans [i, i+1)) as a bool mask.

    A pixel is inside iff its centre is inside the polygon, so sub-pixel vertices are respected.
    """
    mask = np.zeros(shape, bool)
    h, w = shape
    for poly in polygons:
        x0, y0 = np.clip(np.floor(poly.min(0)).astype(int), 0, [w, h])
        x1, y1 = np.clip(np.ceil(poly.max(0)).astype(int), 0, [w, h])
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        centres = np.stack([xs.ravel() + 0.5, ys.ravel() + 0.5], 1)
        mask[y0:y1, x0:x1] |= Polygon(poly).contains_points(centres).reshape(y1 - y0, x1 - x0)
    return mask


def load_label(json_path: Path) -> Label:
    data = json.loads(json_path.read_text())
    shape = (data["imageHeight"], data["imageWidth"])
    by_label: dict[str, list[np.ndarray]] = {"hand": [], "hole": [], "ignore": []}
    for s in data["shapes"]:
        if s.get("shape_type", "polygon") != "polygon":
            raise ValueError(f"{json_path}: only polygons are supported, got {s['shape_type']}")
        if s["label"] not in by_label:
            raise ValueError(f"{json_path}: unknown label {s['label']!r}, expected {list(by_label)}")
        by_label[s["label"]].append(np.asarray(s["points"], np.float64))
    hand = rasterize(by_label["hand"], shape) & ~rasterize(by_label["hole"], shape)
    return Label(json_path.stem, json_path.parent / data["imagePath"], hand,
                 rasterize(by_label["ignore"], shape))


def load_dir(label_dir: Path) -> list[Label]:
    return [load_label(p) for p in sorted(label_dir.glob("*.json"))]
