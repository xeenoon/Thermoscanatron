"""Toolchain hello world: import every dependency and run one tiny thing with each."""

import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
import mediapipe as mp
import numpy as np
import segmentation_models_pytorch as smp
import timm
import torch
import torchvision
import yaml
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def check(name: str, fn) -> None:
    detail = fn()
    print(f"OK  {name:28s} {detail}")


def main() -> None:
    print(f"python {sys.version.split()[0]}")
    for pkg in ["torch", "torchvision", "timm", "segmentation-models-pytorch", "albumentations",
                "executorch", "mediapipe", "opencv-python", "numpy", "pillow"]:
        print(f"    {pkg:28s} {version(pkg)}")
    print()

    img = np.random.randint(0, 255, (64, 64, 3), np.uint8)
    tmp = Path(tempfile.mkdtemp())

    check("numpy / opencv", lambda: cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150).shape)
    check("pillow", lambda: Image.fromarray(img).size)
    check("albumentations", lambda: A.HorizontalFlip(p=1)(image=img)["image"].shape)
    check("mediapipe", lambda: mp.Image(image_format=mp.ImageFormat.SRGB, data=img).width)

    def torch_train_step():
        x, w = torch.randn(4, 3), torch.randn(3, 1, requires_grad=True)
        (x @ w).sum().backward()
        return f"grad {tuple(w.grad.shape)}, cuda={torch.cuda.is_available()}"
    check("torch autograd", torch_train_step)
    check("torchvision", lambda: torchvision.transforms.functional.resize(torch.rand(3, 64, 64), [32]).shape)
    check("timm", lambda: timm.create_model("resnet18", pretrained=False, num_classes=2)(torch.rand(1, 3, 64, 64)).shape)
    check("segmentation_models_pytorch",
          lambda: smp.Unet("resnet18", encoder_weights=None, classes=1).eval()(torch.rand(1, 3, 64, 64)).shape)

    def executorch_roundtrip():
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2)).eval()
        x = torch.randn(1, 4)
        program = to_edge_transform_and_lower(torch.export.export(model, (x,)),
                                              partitioner=[XnnpackPartitioner()]).to_executorch()
        pte = tmp / "hello.pte"
        pte.write_bytes(program.buffer)
        out = Runtime.get().load_program(pte).load_method("forward").execute([x])[0]
        diff = (out - model(x)).abs().max().item()
        assert diff < 1e-5, diff
        return f"export -> .pte -> run, max diff {diff:.1e}"
    check("executorch (xnnpack)", executorch_roundtrip)

    def logging_stack():
        SummaryWriter(tmp / "tb").close()
        matplotlib.use("Agg")
        return f"tensorboard, matplotlib {matplotlib.get_backend()}, yaml {yaml.safe_load('a: 1')}, " \
               f"tqdm {sum(tqdm(range(3), disable=True))}"
    check("logging / config", logging_stack)

    print("\nHELLO WORLD OK")


if __name__ == "__main__":
    main()
