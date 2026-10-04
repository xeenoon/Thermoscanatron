# Android apps

Two native Android apps connect on-device computer vision to the USB thermal sensor:

| Module | Application ID | Purpose |
| --- | --- | --- |
| `app/` | `com.euhack.hello` | Hand/skin segmentation, camera calibration and fused thermal viewing |
| `solar/` | `com.euhack.solar` | Solar-cell tracking, temperature mapping and anomaly overlays |

## Setup

Install JDK 17 and the Android SDK. From the repository root, enter this directory and set the paths for your machine:

```bash
cd android-app
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$ANDROID_HOME/emulator:$ANDROID_HOME/platform-tools:$PATH"
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
```

## Run the hand app on a phone

Enable USB debugging and connect the phone, then build and install:

```bash
./gradlew :app:assembleDebug
adb -d install -r app/build/outputs/apk/debug/app-debug.apk
adb -d shell am start -n com.euhack.hello/.DemoActivity
```

The hand app runs live hand/skin segmentation. Its **Options** menu provides:

- **Record camera + thermal**: a silent ≤1080p video to `files/videos/hand_<time>.mp4` (see the [ML workflow](../ML/README.md)) plus a
  session in `files/sessions/<time>/` with every analysed frame (crop, mask, timestamps) and every thermal packet on
  the same clock (`SessionRecorder.kt`). `ML/src/segkit/thermal_calib.py` fits a calibration from a session.
- **Dump NO-HAND frames**: diagnostics to `files/diagnostics/session_<time>/` (see `NoHandLogger.kt`).
- **Stream thermal input**: shows only the MLX90640 thermal image (°C range, max, sensor temperature).
- **Calibrate thermal ↔ camera**: follow the guided near, far, upper-left and lower-left hand positions.
  Keep the open hand visible to both cameras. The result reports yaw, pitch and roll in degrees, translation
  in centimetres and approximate uncertainty. **Use it** saves the result to `files/thermal_calib.json`.
- **Fused thermal view** (after calibrating): displays the cameras' overlapping field of view, with the rest
  black. Visible-image edges guide the enlarged thermal overlay to help associate readings with objects.
  The sensor still measures only 32 × 24 thermal pixels. The status line shows the hand's median temperature
  and the colour scale.

## Thermal camera over USB OTG

Plug the QT Py running the [firmware](../firmware/README.md) into the phone using USB-C or an OTG adapter.
The phone powers the board and reads its calibrated `THM2` stream. Grant Android's USB-access prompt when
connecting the board. Plugging it in can also offer to open the app.

The attachment occupies the phone's USB port. Use wireless debugging (`adb pair` and `adb connect`) during
hardware development.

## Camera calibration

The [Kotlin solver](app/src/main/java/com/euhack/hello/ThermalCalibration.kt) corresponds to the
[Python calibration pipeline](../ML/src/segkit/thermal_calib.py). It fits camera rotation, translation,
thermal lens scale and readout mirroring, subject to physical mounting constraints.

1. **Timing:** estimate thermal latency from the correlation between visible hand area and thermal warm area.
   If the capture cannot establish latency, use the recorded default of 225 ms.
2. **Pose:** estimate hand depth from mask area, assuming an open hand of approximately 130 cm². Match the
   resulting 3D hand centroids to thermal centroids using robust Levenberg–Marquardt and multiple starting poses.
3. **Refinement:** use Nelder–Mead to maximise correlation between predicted hand coverage and thermal warmth,
   refining pose and lens scale.

Follow the guided near/far and corner positions. Depth variation supplies the parallax needed to distinguish
translation from rotation. The app also uses tighter offset constraints for a sensor attached to the phone.

```bash
./gradlew :app:testDebugUnitTest
```

The tests cover synthetic mountings, including 30° yaw, 45° roll and a 20 cm offset. Recorded-session comparisons
require the corresponding local calibration data.

## Solar Cells app

The second app (`com.euhack.solar`, launcher **Solar Cells**) shares the hand app's USB thermal-stream and
calibration code. Build, test and install it with:

```bash
./gradlew :solar:assembleDebug :solar:testDebugUnitTest
adb -d install -r solar/build/outputs/apk/debug/solar-debug.apk
```

Start with a panel corner visible in the guide box, then move closer. The tracker carries the row and column
identities between frames.

A small model (`assets/panelseg_small.pte`, 192 px) runs on every camera frame; the larger model (`panelseg.pte`,
384 px) runs on its own thread and corrects it. Between frames, and through motion blur, the grid follows the
gyroscope. With the thermal camera connected and a saved or bundled calibration, the app displays per-cell
temperatures and flags differences greater than 5 °C from the panel mean. These flags include both hotspots
and cold anomalies.

**Rec** saves model inputs to `files/panel_debug/<time>/` for replay with `segkit-panel-track --crops`.
See the [ML workflow](../ML/README.md#per-cell-temperatures). Models are trained with `segkit-panel-train`.

## Run in an emulator

Use the emulator for the hand app's camera/UI workflow; USB thermal hardware requires a physical phone.
Run these commands after the setup above:

```bash
emulator -list-avds
emulator -avd igcap &
adb -e wait-for-device
while [ "$(adb -e shell getprop sys.boot_completed | tr -d '\r')" != "1" ]; do sleep 1; done

./gradlew :app:assembleDebug
adb -e install -r app/build/outputs/apk/debug/app-debug.apk
adb -e shell am start -n com.euhack.hello/.DemoActivity
```

Replace `igcap` with an AVD shown by `emulator -list-avds`.

## Hand inference performance diagnostics

Normal launches run only the small model with four native threads. The larger model, its input buffer, and its
executor are not created. This avoids background inference blocking tracking and resetting the thread pool.
The larger model remains bundled for optional ADB comparisons. `-S` restarts the process so the native thread
pool and model state start fresh.

```bash
adb -d shell am start -S -W -n com.euhack.hello/.DemoActivity \
  --ez perf_diagnostics true --ez perf_background false --ei perf_threads 4
adb -d logcat -s HandPerf:I HandDemo:I HandTwoTier:I ExecuTorch:I
```

`perf_background` defaults to false. `perf_threads=1..8` sets the requested pool size on each loaded module;
0 or omission uses four threads for the small model (and two for the larger model if explicitly enabled).
`perf_diagnostics` defaults to false. To compare against the previous two-model configuration:

```bash
adb -d shell am start -S -W -n com.euhack.hello/.DemoActivity \
  --ez perf_diagnostics true --ez perf_background true
```

To restore the normal small-model configuration while retaining the detailed logs:

```bash
adb -d shell am start -S -W -n com.euhack.hello/.DemoActivity --ez perf_diagnostics true
```

`HandPerf` reports normalisation, input tensor construction, native `forward`, output extraction, and
background input preparation in microseconds. `callerCpuUs` counts only the calling thread's CPU time,
not the native worker threads; compare it with wall time and the background-off control, not with total CPU
usage. `bigBusyAtStart` says whether a background job was outstanding when the frame began. Every 30 frames
it also logs camera dimensions, analysis and sensor timestamp gaps, active recording/fusion modes, and UI
queue delay (not display presentation latency). Existing `HandDemo` stage times remain in milliseconds.

Measured before changing the default on the Galaxy A35, 2026-10-03, with the bundled skin models, live 640×480 camera input, and
recording, dumping and fusion off. Each mode ran for 15 seconds after launch, excluding its first 30
analysed frames. That diagnostic build loaded both models even when background inference was off; the current
build skips loading the disabled larger model. FPS below is measured from consecutive analysis log timestamps;
it is not 1000/frame time.

| Configuration | Samples | Median native small call | Median whole analysis | Observed FPS |
| --- | ---: | ---: | ---: | ---: |
| Background on, shared pool 2 | 71 | 114.65 ms | 138 ms | 6.47 |
| Background off, shared pool 2 | 348 | 30.92 ms | 37 ms | 26.49 |
| Background off, shared pool 4 | 403 | 20.91 ms | 27 ms | 29.99 |
| Background on, shared pool 4 | 121 | 77.86 ms | 94 ms | 10.25 |
| Background on, shared pool 2, repeat | 82 | 115.01 ms | 132 ms | 7.55 |

The background model blocks the foreground inference path. In the repeat run, the small call takes 115 ms
of wall time but only 31.58 ms of caller CPU, versus 30.92 ms wall/30.56 ms caller CPU without background
inference. Input copying is about 0.6–1.4 ms and output extraction about 0.2 ms. The small model reached approximately 30 FPS with four threads in this short test. These short runs do not establish sustained thermal
performance or accuracy with background checking disabled.

Two native-runtime details explain the behaviour:

- ExecuTorch 1.5.1's Android JNI [resets a process-wide thread pool on module construction](https://github.com/pytorch/executorch/blob/v1.5.1/extension/android/jni/jni_layer.cpp#L320-L339).
  Loading the larger model last changes both models to two threads. The device logs explicitly show resets to
  four and then two; the requested counts are not independent budgets.
- Its [Android build enables shared XNNPACK workspace](https://github.com/pytorch/executorch/blob/v1.5.1/tools/cmake/preset/android.cmake#L19),
  and [delegate execution holds the workspace lock](https://github.com/pytorch/executorch/blob/v1.5.1/backends/xnnpack/runtime/XNNPACKBackend.cpp#L158-L189).
  Moving the larger model to a low-priority Java thread does not give it independent native resources.

The chosen default is small-model tracking with four threads; the larger model's presence veto/confirmation is
disabled. Keeping background checking while fixing the blocking would require isolating its native execution
resources (for example, a separate Android process), then repeating the handset measurements. Merely
increasing the shared pool to four threads still gave about 10 FPS in the comparison.

After installing this default, a normal launch with no extras measured **29.94 FPS** over 545 frames after
discarding the first 30 frames (20-second capture). Median reported small inference, including input tensor
construction, was 22 ms; whole analysis was 25 ms. Logs confirmed one pool initialisation at four threads,
no background inference, and no model or crash errors.

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
