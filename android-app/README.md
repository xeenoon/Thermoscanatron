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
adb -e shell am start -n com.euhack.hello/.MainActivity
```

Replace `igcap` with an AVD shown by `emulator -list-avds`.
