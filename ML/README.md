# Machine learning and calibration

Python tools for dataset labelling, model training, camera calibration and solar-cell tracking. Models can run in the browser demo or be exported to the [Android apps](../android-app/README.md) with ExecuTorch.

## Setup

Run the examples below from this directory. Use Python 3.12 and `uv`; video extraction also requires `ffmpeg`.

```bash
cd ML
uv sync --extra cpu
uv run --extra cpu segkit-hello-world
uv run --extra cpu pytest
```

For an NVIDIA GPU, replace `--extra cpu` with `--extra cu130` in both setup and run commands. Keep the same extra on every `uv run` invocation so dependency resolution retains the intended PyTorch and ONNX Runtime builds.

Dependencies and command entry points are defined in [pyproject.toml](pyproject.toml).

## Browser webcam demo

The Gradio/FastRTC server processes camera frames supplied by a laptop or phone browser over WebRTC. Select either the stateful solar-cell tracker or the hand-segmentation model.

```bash
uv sync --extra cpu --extra web
uv run --extra cpu --extra web segkit-panel-web
```

Open <http://127.0.0.1:7860>. The demo prefers training checkpoints and falls back to models bundled with the Android apps. Override the models with `--model` / `SEGKIT_PANEL_MODEL` and `--hand-model` / `SEGKIT_HAND_MODEL`.

For a Hugging Face Space, configure an `HF_TOKEN` Space secret for FastRTC's TURN credentials. The demo processes frames in memory without recording them.

## Hand dataset

Record a video in the hand app using **Options → Record camera + thermal**, then copy the videos from the phone:

```bash
adb pull /sdcard/Android/data/com.euhack.hello/files/videos data/
uv run --extra cpu segkit-extract data/videos --out data/captures --fps 3
uv run --extra cpu segkit-label data/captures --out data/hands_v1
uv run --extra cpu segkit-validate data/hands_v1
```

Frame extraction selects the sharpest image in each one-third-second window. Labelling is resumable; use `--limit 5` for a small trial. The default foreground model is `birefnet-general-lite`; `--model isnet-general-use` selects an alternative.

The dataset contains `images/`, `masks/`, `overlays/` and `index.csv`. Inspect the review sheets and list rejected frame stems in `exclude.txt` before training. The separate [skin labeller](src/segkit/label_skin.py) combines human parsing and hand outlines for skin-model datasets.

## Hand evaluation

Place full-resolution frames in `data/eval/` and annotate them with LabelMe:

- **hand:** the hand outline, cut across the wrist.
- **hole:** background enclosed by the hand.
- **ignore:** the wrist-cut band, excluded from scoring.

```bash
uvx --python 3.12 labelme data/eval --output data/eval --labels hand,hole,ignore
uv run --extra cpu segkit-eval check data/eval
uv run --extra cpu segkit-eval score data/eval runs/predictions --csv runs/scores.csv
```

Replace `runs/predictions` with the prediction directory. Each prediction must be a full-resolution PNG named after its source image, with nonzero pixels marking the hand. Check the overlays in `runs/eval_check/` before scoring. The evaluation target is a 95th-percentile boundary error of at most two pixels; this is a target, not a reported result.

## Thermal-camera calibration

Record **Options → Record camera + thermal**, then copy the matching `files/sessions/SESSION_ID/` directory into `data/thermal_sessions/`. Replace `SESSION_ID` in the example:

```bash
uv run --extra cpu python -m segkit.thermal_calib data/thermal_sessions/SESSION_ID --t-ranges 0-55,60-70
uv run --extra cpu pytest tests/test_thermal_calib.py
```

The time ranges select usable seconds from the recording. Results are written to `runs/thermal_calib/thermal_calib.json` and `check.jpg`, including camera pose, lens scale, latency, uncertainty estimates and a projected-hand preview.

The [Python implementation](src/segkit/thermal_calib.py) documents the model and solver. The phone uses the corresponding [Kotlin implementation](../android-app/app/src/main/java/com/euhack/hello/ThermalCalibration.kt). Move the hand nearer and farther during capture so parallax can distinguish camera offset from rotation.

## Solar-cell labelling and training

The default [panel profile](src/segkit/panel/spec.py) describes four columns and nine rows of half-cut cells, including the alternating diamond pattern and gridline width. The labeller accepts `--spec profile.json` for another layout; verify its labels and physical dimensions before using that profile for temperature mapping.

Capture a panel recording with the hand app, moving from a full-panel view into close-ups and back. Set `panel_session_id` to the recording's timestamp:

```bash
panel_session_id=SESSION_ID
panel_session="data/panel/$panel_session_id"
adb pull "/sdcard/Android/data/com.euhack.hello/files/sessions/$panel_session_id" data/panel/
adb pull "/sdcard/Android/data/com.euhack.hello/files/videos/hand_$panel_session_id.mp4" "$panel_session/video.mp4"
mkdir -p "$panel_session/frames"
ffmpeg -i "$panel_session/video.mp4" -q:v 2 "$panel_session/frames/%05d.jpg"
```

Create `anchors.json` in the session directory with at least four panel correspondences for selected wide views. Diamond intersections are useful landmarks; see `segkit-panel-label --help` for the format.

```bash
uv run --extra cpu segkit-panel-label "$panel_session"
uv run --extra cpu segkit-panel-train "$panel_session" --negatives data/panel/negatives.txt --out runs/panel_v1
uv run --extra cpu segkit-panel-track "$panel_session" --model runs/panel_v1/panelseg.pte --video runs/panel_v1/track.mp4
uv run --extra cpu segkit-panel-track "$panel_session" --oracle
```

Review `panel_labels.npz` and `review/sheet_NN.jpg` before training. Correct anchors or add rejected intervals, then rerun the labeller. The negative-image list must contain suitable panel-free examples. Oracle mode evaluates the tracker using label targets instead of model predictions.

PanelNet predicts cell area, gridlines and periodic within-cell coordinates. The tracker maintains integer row and column identities over time; a close-up of a repeating cell pattern cannot establish its identity by appearance alone.

## Per-cell temperatures

Use the saved calibration with either labelled geometry or tracker output:

```bash
uv run --extra cpu segkit-panel-thermal "$panel_session" --calib runs/thermal_calib/thermal_calib.json
uv run --extra cpu segkit-panel-thermal "$panel_session" --calib runs/thermal_calib/thermal_calib.json --source tracker
```

Outputs include `thermal_cells[_tracker].csv` and `thermal_summary[_tracker].json` / `.jpg`: cell temperatures, differences from the panel mean and anomaly flags.

A hotspot is more than 5 °C above the panel mean. The implementation flags differences greater than 5 °C in either direction, so its flags also include cold anomalies. Thermal-pixel footprints are projected onto the panel plane; only footprints entirely within one cell contribute to that cell's reading. This reduces contamination from boundaries and background.

The [Solar Cells app](../android-app/README.md#solar-cells-app) also records analysis crops for `segkit-panel-track --crops` replay.

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
