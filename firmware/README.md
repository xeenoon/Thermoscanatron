# Thermal-sensor firmware

ESP32-S3 firmware for the QT Py and MLX90640 thermal camera. It reads the sensor, converts measurements to temperatures and streams binary packets to the [Android apps](../android-app/README.md) or [desktop viewer](../test_thermal_stream/README.md).

The project uses ESP-IDF v5.5.5, its FreeRTOS port and Espressif's `led_strip` component. Downloaded dependencies and build output are excluded from Git.

## Build and flash

From the repository root:

```bash
cd firmware
./tools/bootstrap.sh
source dependencies/esp-idf/export.sh
idf.py build
idf.py flash monitor
```

The bootstrap script downloads the pinned ESP-IDF release and installs its ESP32-S3 tools. If multiple serial devices are connected, select the board explicitly:

```bash
idf.py -p /dev/ttyACM0 flash monitor
```

Hardware guides and pinouts are indexed in [docs/README.md](../docs/README.md).

## Temperature conversion

The MLX90640 runs in chess mode at eight subpages per second. Each subpage is converted on the board using the Melexis reference driver in `components/mlx90640_melexis` (Apache-2.0) and the sensor's EEPROM calibration.

Conversion uses emissivity **0.95** and reflected temperature **ambient − 8 °C**. Bad pixels are interpolated. These settings affect the reported apparent object temperatures.

## USB packet format

Each subpage produces one **THM2** packet over USB-Serial-JTAG. Packets are **1,566 bytes**, with multibyte fields encoded little-endian.

| Byte offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 4 | Magic: `THM2` |
| 4 | 1 | Protocol version: 2 |
| 5 | 1 | Subpage: 0 or 1 |
| 6 | 2 | Header length: 28 bytes |
| 8 | 4 | Sequence number |
| 12 | 8 | `esp_timer` timestamp in microseconds |
| 20 | 2 | Payload length: 769 words |
| 22 | 1 | Image width: 32 |
| 23 | 1 | Image height: 24 |
| 24 | 4 | CRC32 of the payload |
| 28 | 1,536 | 768 signed 16-bit pixel temperatures, row-major, in hundredths of a degree Celsius |
| 1,564 | 2 | Signed 16-bit sensor ambient temperature, in hundredths of a degree Celsius |

Each packet contains a merged image: the newly read subpage and the other half from the preceding subpage. The two halves therefore represent successive sensor reads, not a simultaneous full-frame exposure.
