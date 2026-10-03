# ML Run instructions

```bash
cd ML
uv sync --extra cpu          # laptop (CPU torch); GPU box: uv sync --extra cu130
# always pass the same --extra to `uv run`; plain `uv run` installs the CUDA build
uv run segkit-hello-world   # imports every dependency, runs one small op each; ends with "HELLO WORLD OK"
uv run pytest               # unit tests
```

### Hand dataset (phone video -> frames -> rembg labels)

Record with the Android app (`android-app/`, Options ▾ → Record video: silent 1080p video), then:

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

### Thermal ↔ camera calibration

Record with the app (Options ▾ → Record camera + thermal), pull `files/sessions/<time>/` into
`data/thermal_sessions/`, then:

```bash
uv run python -m segkit.thermal_calib data/thermal_sessions/<time> --t-ranges 0-55,60-70   # seconds to use
# -> runs/thermal_calib/thermal_calib.json (pose, lens scale, latency, 1σ) + check.jpg (predicted hand on thermal)
uv run pytest tests/test_thermal_calib.py      # recovers synthetic mountings, e.g. 30° yaw + 45° roll + 20 cm
```

The model and both solver stages are documented in `src/segkit/thermal_calib.py`; the phone runs the same
algorithm (`android-app/.../ThermalCalibration.kt`). The thermal sensor is the 55°×35° MLX90640-BAB.

### Solar panel cells (segment + track which cell is which)

Panel profile (`src/segkit/panel/spec.py`): cell lattice (4 x 9 half-cut mono cells), which row boundaries carry
the wafer-corner diamonds, gridline width. Pass `--spec profile.json` to the tools for another panel.

Record with the app (Options ▾ → Record), walking from far away (whole panel in view) to close-ups and back, then:

```bash
S=data/panel/<session>
adb pull /sdcard/Android/data/com.euhack.hello/files/sessions/<session> data/panel/
adb pull /sdcard/Android/data/com.euhack.hello/files/videos/hand_<session>.mp4 $S/video.mp4
mkdir -p $S/frames && ffmpeg -i $S/video.mp4 -q:v 2 $S/frames/%05d.jpg
# $S/anchors.json: a few far-away frames with 4+ panel points each (diamonds are easiest), see segkit-panel-label -h
uv run segkit-panel-label $S                 # -> $S/panel_labels.npz + $S/review/sheet_NN.jpg (check them;
                                             #    add "reject" ranges / more anchors and re-run)
uv run segkit-panel-train $S --negatives data/panel/negatives.txt --out runs/panel_v1   # -> panelseg.pte
uv run segkit-panel-track $S --model runs/panel_v1/panelseg.pte --video runs/panel_v1/track.mp4
uv run segkit-panel-track $S --oracle        # tracker alone, fed the label targets
# per-cell temperatures + hotspots, thermal pose from segkit.thermal_calib (runs/thermal_calib/thermal_calib.json)
uv run segkit-panel-thermal $S --calib runs/thermal_calib/thermal_calib.json                   # label geometry
uv run segkit-panel-thermal $S --calib runs/thermal_calib/thermal_calib.json --source tracker  # live pipeline
# -> $S/thermal_cells[_tracker].csv: per frame panel average, 36 cell temps, hotspot cells
#    $S/thermal_summary[_tracker].{json,jpg}: per cell temp + offset from the panel average, hotspot flags
```

Hotspot = a cell more than 5 °C above or below the panel average (`HOTSPOT_DELTA_C`). Each thermal pixel's
footprint is cast onto the panel plane (pose from the panel homography and the phone camera's intrinsics); only
pixels landing wholly inside one cell count for that cell, so frame and background never leak in.

The model predicts, per pixel: cell area, gridlines, and where inside its cell the pixel is
(sin/cos of u with a one-cell period, of v with a two-row period). It never predicts the cell index: up close
every cell looks the same. The tracker (`segkit.panel.track`) supplies that from memory: it locks on when
enough of the panel outline is in view, then carries the integer cell coordinates from frame to frame
while the model pins the position inside the cell, so the label stays right when zoomed onto a single cell.

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

## References

- F. Hong, J. Song, H. Meng, R. Wang, F. Fang, G. Zhang, "A novel framework on intelligent detection for module
  defects of PV plant combining the visible and infrared images", *Solar Energy* 236 (2022) 406–416,
  [doi:10.1016/j.solener.2022.03.018](https://doi.org/10.1016/j.solener.2022.03.018). The architecture this
  follows: the visible camera supplies geometry/segmentation, the low-resolution IR camera the temperatures,
  joined by a calibrated mapping.
- K. He, J. Sun, X. Tang, "Guided Image Filtering", *IEEE TPAMI* 35(6) (2013) 1397–1409,
  [doi:10.1109/TPAMI.2012.213](https://doi.org/10.1109/TPAMI.2012.213). Used to upsample the 32×24 temperatures
  along the camera's edges (`android-app/.../FusionRenderer.kt`).
- J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele, "Joint Bilateral Upsampling", *ACM TOG* 26(3) (2007) 96,
  [doi:10.1145/1276377.1276497](https://doi.org/10.1145/1276377.1276497). The same guided-upsampling idea; the guided
  filter is used because it runs in O(pixels).
