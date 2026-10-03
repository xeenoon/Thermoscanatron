"""Serve the panel scanner as a browser webcam app with Gradio and FastRTC."""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from segkit.datasets.hands import IMAGENET_MEAN, IMAGENET_STD
from segkit.models.handseg import HandSegNet, ProbabilityHead
from segkit.panel.spec import PanelSpec
from segkit.panel.track import BOX_FRACTION, Model, PanelTracker, centre_crop, draw
from segkit.test_set import largest_blob

PANEL_MODE = "Solar panel"
HAND_MODE = "Hand recognition"


@dataclass
class Session:
    tracker: PanelTracker
    mode: str = PANEL_MODE
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)


class HandModel:
    """HandSeg deployment wrapper accepting the same .pt and .pte artifacts as the native app."""

    def __init__(self, path: Path):
        if path.suffix == ".pte":
            from executorch.runtime import Runtime

            method = Runtime.get().load_program(path).load_method("forward")
            self.run = lambda x: method.execute([x])
        else:
            net = HandSegNet()
            net.load_state_dict(torch.load(path, map_location="cpu"))
            head = ProbabilityHead(net).eval()
            self.run = lambda x: head(x)

    @torch.no_grad()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        x = (rgb.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        mask, present = self.run(tensor)
        return mask[0, 0].numpy(), float(present[0, 0])


class Scanner:
    """Shared models with independent panel-tracking state for each WebRTC connection."""

    def __init__(self, panel_path: Path, panel_size: int, hand_path: Path, hand_size: int):
        self.panel_size = panel_size
        self.hand_size = hand_size
        self.spec = PanelSpec()
        self.panel_model = Model(panel_path, panel_size)
        self.hand_model = HandModel(hand_path)
        self.inference_lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()

    def _session(self, connection_id: str) -> Session:
        now = time.monotonic()
        with self.sessions_lock:
            # WebRTC does not currently expose a disconnect callback here. Bound stale state instead.
            if len(self.sessions) > 64:
                self.sessions = {key: value for key, value in self.sessions.items() if now - value.touched < 1800}
            session = self.sessions.get(connection_id)
            if session is None:
                session = Session(PanelTracker(self.spec, self.panel_size, klt=True))
                self.sessions[connection_id] = session
            session.touched = now
            return session

    @staticmethod
    def _place_crop(frame_bgr: np.ndarray, annotated: np.ndarray) -> np.ndarray:
        """Put a model-sized annotated crop back into the centre of the full camera frame."""
        output = frame_bgr.copy()
        height, width = output.shape[:2]
        side = max(1, int(BOX_FRACTION * min(height, width)))
        left, top = (width - side) // 2, (height - side) // 2
        output[top : top + side, left : left + side] = cv2.resize(
            annotated, (side, side), interpolation=cv2.INTER_LINEAR
        )
        cv2.rectangle(output, (left, top), (left + side - 1, top + side - 1), (255, 255, 255), 2)
        return output

    @staticmethod
    def _draw_hand(crop_bgr: np.ndarray, mask: np.ndarray, present: float) -> np.ndarray:
        output = crop_bgr.copy()
        hand = largest_blob(mask > 0.5) if present > 0.5 else np.zeros_like(mask, dtype=bool)
        if hand.any():
            green = np.zeros_like(output)
            green[:, :, 1] = 255
            blended = cv2.addWeighted(output, 0.65, green, 0.35, 0)
            output[hand] = blended[hand]
            contours, _ = cv2.findContours(hand.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, (0, 255, 0), 3)
        label = "HAND" if hand.any() else "NO HAND"
        colour = (0, 255, 0) if hand.any() else (255, 255, 255)
        cv2.putText(output, f"{label}  {present:.2f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
        return output

    def process(self, frame_bgr: np.ndarray, mode: str = PANEL_MODE) -> np.ndarray:
        """Run one webcam frame and return a full-frame annotated BGR image."""
        from fastrtc import get_current_context

        connection_id = get_current_context().webrtc_id
        session = self._session(connection_id)
        with session.lock:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if mode == HAND_MODE:
                crop_rgb, _ = centre_crop(frame_rgb, self.hand_size)
                with self.inference_lock:
                    mask, present = self.hand_model(crop_rgb)
                annotated = self._draw_hand(cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), mask, present)
            else:
                if session.mode != PANEL_MODE:
                    session.tracker = PanelTracker(self.spec, self.panel_size, klt=True)
                crop_rgb, _ = centre_crop(frame_rgb, self.panel_size)
                with self.inference_lock:
                    dense, present = self.panel_model(crop_rgb)
                result = session.tracker.step(crop_rgb, dense, present)
                crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                annotated = draw(crop_bgr, result, self.spec, dense, label=None)
            session.mode = mode
        return self._place_crop(frame_bgr, annotated)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_panel_model() -> Path:
    """Prefer a training checkpoint, falling back to the model already shipped in Android."""
    root = repo_root()
    candidates = [
        Path(os.environ["SEGKIT_PANEL_MODEL"]) if "SEGKIT_PANEL_MODEL" in os.environ else None,
        root / "ML/runs/panel_v1/best.pt",
        root / "android-app/solar/src/main/assets/panelseg_small.pte",
        Path("models/panelseg_small.pte"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("No panel model found; pass --model or set SEGKIT_PANEL_MODEL")


def default_hand_model() -> Path:
    root = repo_root()
    candidates = [
        Path(os.environ["SEGKIT_HAND_MODEL"]) if "SEGKIT_HAND_MODEL" in os.environ else None,
        root / "ML/runs/handseg_v3/best.pt",
        root / "android-app/app/src/main/assets/handseg.pte",
        Path("models/handseg.pt"),
        Path("models/handseg.pte"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("No hand model found; pass --hand-model or set SEGKIT_HAND_MODEL")


def build_demo(scanner: Scanner):
    import gradio as gr
    from fastrtc import WebRTC, get_cloudflare_turn_credentials_async

    # Direct peer connections work locally. A deployed Space uses its HF_TOKEN secret for TURN credentials.
    rtc_configuration = get_cloudflare_turn_credentials_async if os.environ.get("HF_TOKEN") else None
    css = """
    .topbar { align-items: flex-start; gap: 1rem; }
    .topbar-title { flex: 1 1 auto; }
    .topbar > .form { flex: 0 0 240px !important; width: 240px; margin-left: auto; overflow: visible; }
    .mode-picker { width: 100%; max-width: 240px; }
    .scan { max-width: 900px; margin: 0 auto; }
    .scan video { object-fit: contain !important; }
    @media (max-width: 640px) {
        .topbar > .form { flex-basis: 180px !important; width: 180px; }
        .mode-picker { max-width: 180px; }
    }
    """
    with gr.Blocks(title="Solar Panel Scanner", css=css) as demo:
        with gr.Row(elem_classes=["topbar"]):
            gr.Markdown("# Live Vision Demo", elem_classes=["topbar-title"])
            mode = gr.Dropdown(
                [PANEL_MODE, HAND_MODE],
                value=PANEL_MODE,
                label="Detection mode",
                interactive=True,
                elem_classes=["mode-picker"],
                min_width=180,
                scale=0,
            )
        gr.Markdown(
            "Choose a detection mode, click the camera to grant access, then press **Start detection**. "
            "Frames are processed by the server and are not saved."
        )
        camera = WebRTC(
            label="Live detection",
            modality="video",
            mode="send-receive",
            rtc_configuration=rtc_configuration,
            elem_classes=["scan"],
            mirror_webcam=False,
            full_screen=False,
            button_labels={
                "start": "Start detection",
                "stop": "Stop detection",
                "waiting": "Connecting…",
            },
        )
        camera.stream(
            fn=scanner.process,
            inputs=[camera, mode],
            outputs=[camera],
            concurrency_limit=8,
            time_limit=900,
        )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None, help="Panel .pt checkpoint or ExecuTorch .pte")
    parser.add_argument("--hand-model", type=Path, default=None, help="HandSeg .pt checkpoint or ExecuTorch .pte")
    parser.add_argument("--size", type=int, default=int(os.environ.get("SEGKIT_PANEL_SIZE", "192")))
    parser.add_argument("--hand-size", type=int, default=int(os.environ.get("SEGKIT_HAND_SIZE", "384")))
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", help="Create a temporary public Gradio link")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel_path = args.model or default_panel_model()
    hand_path = args.hand_model or default_hand_model()
    print(f"Loading panel model {panel_path} at {args.size}x{args.size}")
    print(f"Loading hand model {hand_path} at {args.hand_size}x{args.hand_size}")
    scanner = Scanner(panel_path, args.size, hand_path, args.hand_size)
    build_demo(scanner).launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
