# QTPI Firmware

Uses ESP-IDF v5.5.5, its FreeRTOS port, and Espressif's `led_strip` component.
Downloaded dependencies and build output are ignored by Git.

```sh
./tools/bootstrap.sh
source dependencies/esp-idf/export.sh
idf.py build
idf.py flash monitor
```
