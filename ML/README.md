# ML Run instructions

```bash
cd ML
uv sync
uv run segkit-hello-world   # imports every dependency, runs one small op each; ends with "HELLO WORLD OK"
```

## Dependencies

Python 3.12 (managed by `uv`), CPU PyTorch wheels from `download.pytorch.org/whl/cpu`.

- torch, torchvision
- timm
- segmentation-models-pytorch
- albumentations
- executorch
- mediapipe
- opencv-python, numpy, pillow
- tensorboard, tqdm, pyyaml, matplotlib
- jupyter (dev)
