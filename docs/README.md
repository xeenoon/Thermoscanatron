# Hardware documentation

Local reference manuals for the thermal attachment's microcontroller and infrared sensor. Use these documents
for wiring, pin assignments, electrical limits and sensor communication. Build instructions are in the
[firmware README](../firmware/README.md); phone integration is covered by the [Android README](../android-app/README.md).

## QT Py ESP32-S3 (ADA5700)

- [Adafruit product guide](qt-py-esp32-s3/adafruit-qt-py-esp32-s3-guide.pdf) — pinouts, power, peripherals, schematic, and fabrication drawing
- [Adafruit pinout](qt-py-esp32-s3/adafruit-qt-py-esp32-s3-pinout.pdf) — one-page board pin map
- [ESP32-S3 datasheet](qt-py-esp32-s3/esp32-s3-datasheet.pdf) — chip specifications and GPIO functions
- [ESP32-S3 technical reference manual](qt-py-esp32-s3/esp32-s3-technical-reference-manual.pdf) — peripheral and register reference

## MLX90640 thermal camera (ADA4407)

The project uses the MLX90640-BAB variant, with a nominal 55° × 35° field of view.

- [Adafruit product guide](mlx90640/adafruit-mlx90640-breakout-guide.pdf) — breakout pinouts, wiring, schematic, and fabrication drawing
- [Melexis MLX90640 datasheet](mlx90640/mlx90640-datasheet.pdf) — sensor specifications, I2C protocol, registers, and EEPROM layout
- [Melexis MLX90640 driver guide](mlx90640/mlx90640-driver.pdf) — reference-driver API and measurement flow

## Upstream sources

- [Adafruit QT Py ESP32-S3 guide](https://learn.adafruit.com/adafruit-qt-py-esp32-s3)
- [QT Py ESP32-S3 PCB files](https://github.com/adafruit/Adafruit-QT-Py-ESP32-S3-PCB)
- [Espressif technical documentation](https://www.espressif.com/en/support/documents/technical-documents)
- [Adafruit MLX90640 guide](https://learn.adafruit.com/adafruit-mlx90640-ir-thermal-camera)
- [Melexis MLX90640 datasheet](https://www.melexis.com/en/documents/documentation/datasheets/datasheet-mlx90640)
- [Melexis reference driver](https://github.com/melexis/mlx90640-library)
