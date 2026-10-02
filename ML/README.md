# ML Run instructions

```bash
cd ML
uv sync --extra cpu          # laptop (CPU torch); GPU box: uv sync --extra cu130
# always pass the same --extra to `uv run`; plain `uv run` installs the CUDA build
uv run segkit-hello-world   # imports every dependency, runs one small op each; ends with "HELLO WORLD OK"
uv run pytest               # unit tests
```

### Hand dataset (phone video -> frames -> rembg labels)

Record with the Android app (`android-app/`, Record/Stop: silent 1080p video), then:

```bash
adb pull /sdcard/Android/data/com.euhack.hello/files/videos data/      # -> data/videos/*.mp4
uv run segkit-extract data/videos --out data/captures --fps 3          # sharpest frame per 1/3 s window
uv run segkit-label data/captures --out data/hands_v1                  # resumable; --limit 5 to try first
uv run segkit-validate data/hands_v1                                   # checks + review/sheet_NN.jpg
# data/hands_v1/{images,masks,overlays}/ + index.csv (flags: area, extra_blobs, blurry)
# data/hands_v1/exclude.txt: stems rejected in manual review (skipped by training)
# --model isnet-general-use is ~10x faster on CPU, birefnet-general-lite (default) has better edges
```

### Eval set (hand outlines)

Put full-res frames in `data/eval/`, then label them with polygons in LabelMe:
`hand` = outline cut straight across the wrist, `hole` = background enclosed by the hand,
`ignore` = a band over the wrist cut (not scored). Zoom in and place vertices on the edge.

```bash
uvx --python 3.12 labelme data/eval --output data/eval --labels hand,hole,ignore
uv run segkit-eval check data/eval                      # overlays in runs/eval_check/ to verify labels
uv run segkit-eval score data/eval <pred_dir> --csv runs/scores.csv
# <pred_dir>/<image stem>.png, full-res, nonzero = hand. Target: p95 boundary error <= 2px
```

## Dependencies

Python 3.12 (managed by `uv`), PyTorch from `download.pytorch.org/whl/cpu` or `/whl/cu130` (extra `cpu` / `cu130`).

- torch, torchvision
- timm
- segmentation-models-pytorch
- albumentations
- executorch
- mediapipe
- opencv-python, numpy, pillow
- tensorboard, tqdm, pyyaml, matplotlib
- rembg[cpu] (background removal labels)
- jupyter, pytest (dev)
- labelme (run via `uvx`, not installed in the env)
