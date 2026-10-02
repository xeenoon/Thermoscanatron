# Thermal stream test viewer

Displays the raw 32×24 MLX90640 pixel words streamed by the QT Py firmware. A minimal C reader owns the serial device; Node only serves the image.

```sh
npm install
npm run simulate
```

Open <http://localhost:3000>. The simulator validates the full binary parsing and rendering path without hardware.

With the board connected at `/dev/ttyACM0`:

```sh
npm start
```

Use `npm test` for parser tests and `npm run validate` for a fragmented, noisy simulated-stream check.
