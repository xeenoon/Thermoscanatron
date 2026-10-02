# QTPI Firmware

Uses ESP-IDF v5.5.5, its FreeRTOS port, and Espressif's `led_strip` component.
Downloaded dependencies and build output are ignored by Git.

```sh
./tools/bootstrap.sh
source dependencies/esp-idf/export.sh
idf.py build
idf.py flash monitor
```

## Thermal stream

The MLX90640 runs in chess mode at 8 subpages/s. Each subpage is converted to object temperatures on the
board with the Melexis reference driver (`components/mlx90640_melexis`, Apache-2.0) using the sensor's
EEPROM calibration (emissivity 0.95, reflected temperature = ambient − 8 °C), and bad pixels are interpolated.

Every subpage goes out over USB-Serial-JTAG as one `THM2` packet (little endian, 1566 bytes):

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `THM2` |
| 4 | 1 | version (2) |
| 5 | 1 | subpage (0/1) |
| 6 | 2 | header bytes (28) |
| 8 | 4 | sequence |
| 12 | 8 | `esp_timer` microseconds |
| 20 | 2 | payload words (769) |
| 22 | 1 | width (32) |
| 23 | 1 | height (24) |
| 24 | 4 | CRC32 of the payload |
| 28 | 1536 | 768 × int16 pixel temperatures, centi-°C, row-major |
| 1564 | 2 | int16 sensor ambient temperature, centi-°C |

Every packet carries the full merged image (the subpage that was just read, plus the other half from the previous subpage).
Readers: `test_thermal_stream/` (laptop) and `android-app/` (phone, USB OTG; the phone powers the board).
