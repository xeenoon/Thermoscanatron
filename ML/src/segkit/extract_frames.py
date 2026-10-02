"""Extract training frames from capture videos.

segkit-extract data/videos --out data/captures --fps 3

Splits each video into windows of 1/fps seconds and keeps the sharpest frame of each window
(Laplacian variance), so frames blurred by hand/phone motion are mostly skipped.
Writes <out>/<video stem>/<video stem>_f<frame>.jpg, which `segkit-label` picks up.
Videos with a finished output dir (.done marker) are skipped.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

SHARPNESS_SIDE = 480  # sharpness is measured on a downscaled copy; enough to rank blur


def sharpness(bgr: np.ndarray) -> float:
    scale = SHARPNESS_SIDE / max(bgr.shape[:2])
    gray = cv2.cvtColor(cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def extract(video: Path, out_dir: Path, fps: float, min_sharpness: float) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(video))  # applies the phone's rotation metadata (CAP_PROP_ORIENTATION_AUTO)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    window = max(1, round(src_fps / fps))
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = skipped = idx = 0
    best: tuple[float, int, np.ndarray] | None = None

    def flush():
        nonlocal kept, skipped
        if best is None:
            return
        score, frame_idx, frame = best
        if score < min_sharpness:
            skipped += 1
            return
        cv2.imwrite(str(out_dir / f"{video.stem}_f{frame_idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        s = sharpness(frame)
        if best is None or s > best[0]:
            best = (s, idx, frame)
        idx += 1
        if idx % window == 0:
            flush()
            best = None
    flush()
    cap.release()
    (out_dir / ".done").touch()
    return kept, skipped


def main() -> None:
    p = argparse.ArgumentParser(prog="segkit-extract")
    p.add_argument("videos", type=Path, help="a video file or a directory of *.mp4")
    p.add_argument("--out", type=Path, default=Path("data/captures"))
    p.add_argument("--fps", type=float, default=3.0, help="frames kept per second of video")
    p.add_argument("--min-sharpness", type=float, default=0.0,
                   help="drop a window's best frame if its Laplacian variance is below this")
    args = p.parse_args()

    videos = [args.videos] if args.videos.is_file() else sorted(args.videos.rglob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 files in {args.videos}")
    for v in videos:
        out_dir = args.out / v.stem
        if (out_dir / ".done").exists():
            print(f"{v.name}: already extracted")
            continue
        kept, skipped = extract(v, out_dir, args.fps, args.min_sharpness)
        print(f"{v.name}: kept {kept} frames, dropped {skipped} below sharpness -> {out_dir}")


if __name__ == "__main__":
    main()
