# Android app

## Run

```bash
cd android-app
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk

emulator -list-avds
emulator -avd igcap &
adb -e wait-for-device
while [ "$(adb -e shell getprop sys.boot_completed | tr -d '\r')" != "1" ]; do sleep 1; done

./gradlew assembleDebug
adb -e install -r app/build/outputs/apk/debug/app-debug.apk
adb -e shell am start -n com.euhack.hello/.DemoActivity
```

Replace `igcap` with an AVD shown by `emulator -list-avds`.

## Run on a USB phone

```bash
./gradlew assembleDebug
adb -d install -r app/build/outputs/apk/debug/app-debug.apk
adb -d shell am start -n com.euhack.hello/.DemoActivity
```

The app runs the live hand-outline demo (HandSegNet v3). Its **Options ▾** dropdown:

- **Record camera + thermal**: a silent ≤1080p video to `files/videos/hand_<time>.mp4` (see `ML/README.md`) plus a
  session in `files/sessions/<time>/` with every analysed frame (crop, mask, timestamps) and every thermal packet on
  the same clock (`SessionRecorder.kt`). `ML/src/segkit/thermal_calib.py` fits a calibration from a session.
- **Dump NO-HAND frames**: diagnostics to `files/diagnostics/session_<time>/` (see `NoHandLogger.kt`).
- **Stream thermal input**: shows only the MLX90640 thermal image (°C range, max, sensor temperature).
- **Calibrate thermal ↔ camera**: "Hold your hand up inside the box to calibrate" — 5 s of the hand seen by both
  cameras (progress bar), then a popup with where the thermal camera sits relative to the phone camera: yaw, pitch,
  roll in degrees and x, y, z in cm, with rough 1σ. **Use it** saves it to `files/thermal_calib.json`.
- **Fused thermal view** (after calibrating): only the part of the camera image the thermal camera also sees,
  the rest black; temperatures upsampled to camera resolution along the camera's edges, colour-mapped and shaded
  by the camera image with its edges drawn in, so each temperature reads off the object it belongs to. The
  status line shows the hand's temperature (median under the hand mask) and the colour scale.

## Thermal ↔ camera calibration

Nothing about how the sensor is taped on is assumed: any roll, large yaw/pitch, tens of cm of offset, mirrored
or not (`ThermalCalibration.kt`, a port of `ML/src/segkit/thermal_calib.py` where the model is documented).

1. **Pose from points**: each frame's hand centroid, at a depth estimated from the mask area (an open hand is
   ~130 cm²), is a 3D point; the warm blob's centroid is its thermal image. Robust Levenberg–Marquardt from a grid
   of starting rotations (all rolls, yaw/pitch ±60°), for both mirror hypotheses; the thermal latency is the one
   with the smallest reprojection error.
2. **Silhouette refinement**: Nelder–Mead maximising the correlation between each thermal pixel's predicted hand
   fraction (its rays hit the hand plane and are looked up in the hand mask) and its warmth; also fits the lens
   scale.

Translation is told apart from rotation by parallax, so move the hand nearer and further while calibrating. Unit
tests (`./gradlew testDebugUnitTest`) recover synthetic mountings such as 30° yaw + 45° roll + 20 cm offset, and
match the Python fit on the recorded session when `ML/data/thermal_sessions/` is present.

## Thermal camera over USB OTG

Plug the QT Py (running `firmware/`) into the phone with a USB-C to USB-C cable (or an OTG adapter).
The phone powers the board and reads its calibrated `THM2` temperature stream from the USB-Serial-JTAG port.
The first time, Android asks to allow USB access; plugging in also offers to open the app.
The phone's USB port is then taken, so use wireless debugging (`adb pair` / `adb connect`) while developing.

## References

- F. Hong, J. Song, H. Meng, R. Wang, F. Fang, G. Zhang, "A novel framework on intelligent detection for module
  defects of PV plant combining the visible and infrared images", *Solar Energy* 236 (2022) 406–416,
  [doi:10.1016/j.solener.2022.03.018](https://doi.org/10.1016/j.solener.2022.03.018). The architecture this
  follows: the visible camera supplies geometry/segmentation, the low-resolution IR camera the temperatures,
  joined by a calibrated mapping.
- K. He, J. Sun, X. Tang, "Guided Image Filtering", *IEEE TPAMI* 35(6) (2013) 1397–1409,
  [doi:10.1109/TPAMI.2012.213](https://doi.org/10.1109/TPAMI.2012.213). Used to upsample the 32×24 temperatures
  along the camera's edges (`FusionRenderer.kt`).
- J. Kopf, M. F. Cohen, D. Lischinski, M. Uyttendaele, "Joint Bilateral Upsampling", *ACM TOG* 26(3) (2007) 96,
  [doi:10.1145/1276377.1276497](https://doi.org/10.1145/1276377.1276497). The same guided-upsampling idea; the guided
  filter is used because it runs in O(pixels).

## Solar Cells app (`solar/`)

A second app (`com.euhack.solar`, launcher "Solar Cells") next to the hand demo. It reuses the hand app's
thermal USB stream and calibration code as sources from `app/`.

```bash
./gradlew :solar:assembleDebug :solar:testDebugUnitTest
adb -d install -r solar/build/outputs/apk/debug/solar-debug.apk
```

Point it at the panel with a corner in the box to lock on, then move in: it keeps the (row, col) cell numbers.
A small model (`assets/panelseg_small.pte`, 192 px) runs on every camera frame; the big one (`panelseg.pte`,
384 px) runs on its own thread and corrects it. Between frames, and through motion blur, the grid follows the
gyroscope. With the thermal camera plugged in it shows per-cell temperatures and hotspots (more than 5 °C from
the panel average). **Rec** saves the model input to `files/panel_debug/<time>/` for replay with
`segkit-panel-track --crops` (see `ML/README.md`). Models come from `segkit-panel-train`.
