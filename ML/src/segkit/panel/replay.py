"""Read old thermal crops and versioned solar diagnostics without confusing predictions with labels."""
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class RecordedFrame:
    path: Path
    frame: int
    sensor_ns: int | None
    fields: dict
    H: np.ndarray | None
    mask: np.ndarray | None


def load_recording(session: Path, size: int) -> list[RecordedFrame]:
    meta_path = session / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    csv_path = session / "frames.csv"
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    keyed = {int(row.get("frame", i)): row for i, row in enumerate(rows)}
    if len(keyed) != len(rows):
        raise ValueError("duplicate frame IDs in frames.csv")
    paths = []
    for directory in (session / "crops", session, session / "frames"):
        paths = sorted(p for p in directory.glob("*.jpg") if p.stem.isdigit())
        if paths:
            break
    if not paths:
        raise ValueError(f"No recorded crops in {session}")
    out = []
    for path in paths:
        frame = int(path.stem)
        if frame not in keyed:
            raise ValueError(f"Crop {path.name} has no CSV row")
        fields = keyed[frame]
        ts = fields.get("sensor_ns") or fields.get("sensor_ts_ns")
        values = [fields.get(f"h{i}", "") for i in range(9)]
        H = None
        if any(values):
            if not all(values):
                raise ValueError(f"Incomplete homography at frame {frame}")
            H = np.array(values, float).reshape(3, 3)
            if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                raise ValueError(f"Invalid homography at frame {frame}")
            native = float(meta.get("tracker_size", 384))
            H = np.diag([size / native, size / native, 1.]) @ H
        mask = None
        if fields.get("mask_file"):
            mask = cv2.imread(str(session / fields["mask_file"]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Missing mask at frame {frame}")
            mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST) > 127
        out.append(RecordedFrame(path, frame, int(ts) if ts else None, fields, H, mask))
    return out
