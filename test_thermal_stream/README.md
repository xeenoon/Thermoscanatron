# Thermal stream viewer

A browser viewer for calibrated **32 × 24 MLX90640 temperatures** streamed by the [QT Py firmware](../firmware/README.md). A native C reader handles the serial device; the Node.js server parses packets and sends temperature frames to the browser.

## Requirements

- Node.js 20 or later and npm.
- CMake and a C compiler for the hardware serial reader.
- A QT Py running the firmware for live measurements.

The simulator runs without the sensor or native reader.

## Try the simulator

From the repository root:

```bash
cd test_thermal_stream
npm install
npm run simulate
```

Open <http://localhost:3000>. Simulated THM2 packets exercise the JavaScript parser and browser display without hardware.

## Read the sensor

Connect the board over USB, list the serial ports, and start the viewer:

```bash
npm run list-ports
npm start
```

The default device is `/dev/ttyACM0`. `npm start` builds the native reader automatically. To choose another device or HTTP port:

```bash
npm start -- --serial /dev/ttyACM1 --http-port 3001
```

The HTTP port also accepts the `PORT` environment variable. Ensure your user has access to the selected serial device.

## Validation

```bash
npm test
npm run validate
```

The tests cover packet parsing. The validation script feeds a fragmented, noisy simulated stream through the parser. See the [firmware packet specification](../firmware/README.md#usb-packet-format) for field definitions.
