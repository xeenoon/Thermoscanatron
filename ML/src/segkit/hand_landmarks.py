"""MediaPipe Hand Landmarker: is there a hand, and where are its 21 landmarks."""

import sys
from pathlib import Path

# mediapipe.tasks imports its audio module, which pulls in sounddevice -> PortAudio -> JACK/PipeWire.
# On PipeWire desktops that gets the process SIGKILLed during HandLandmarker init. We never use audio.
sys.modules.setdefault("sounddevice", None)

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "assets" / "hand_landmarker.task"
DETECT_SIDE = 640  # detection runs on a downscaled copy; landmarks come back in full-res pixels


class HandLandmarks:
    def __init__(self, model_path: Path = DEFAULT_MODEL, num_hands: int = 2, min_confidence: float = 0.3):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def detect(self, rgb: np.ndarray) -> list[np.ndarray]:
        """One [21, 2] float array of full-res pixel coords per detected hand (empty list = no hand)."""
        h, w = rgb.shape[:2]
        scale = DETECT_SIDE / max(h, w)
        small = np.ascontiguousarray(cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        result = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=small))
        return [np.array([[p.x * w, p.y * h] for p in hand], np.float32) for hand in result.hand_landmarks]
