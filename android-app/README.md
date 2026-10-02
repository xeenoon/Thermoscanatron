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

The app runs the live hand-outline demo. Its **Options ▾** dropdown has three toggles:

- **Record video**: a silent ≤1080p, 20 Mbps video to
  `/sdcard/Android/data/com.euhack.hello/files/videos/hand_<time>.mp4`. See `ML/README.md` to pull and label them.
- **Dump NO-HAND frames**: diagnostics to `files/diagnostics/session_<time>/` (see `NoHandLogger.kt`).
- **Stream thermal input**: releases the camera and shows only the MLX90640 thermal image (°C range, max, sensor temperature).

## Thermal camera over USB OTG

Plug the QT Py (running `firmware/`) into the phone with a USB-C to USB-C cable (or an OTG adapter).
The phone powers the board and reads its calibrated `THM2` temperature stream from the USB-Serial-JTAG port.
The first time, Android asks to allow USB access; plugging in also offers to open the app.
The phone's USB port is then taken, so use wireless debugging (`adb pair` / `adb connect`) while developing.
