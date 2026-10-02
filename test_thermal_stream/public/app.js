const canvas = document.querySelector('#thermal');
const context = canvas.getContext('2d');
const readout = document.querySelector('#readout');
const sensorCanvas = document.createElement('canvas');
const sensorContext = sensorCanvas.getContext('2d');
sensorCanvas.width = 32;
sensorCanvas.height = 24;

function heatColor(value) {
  const stops = [
    [0, 0, 80],
    [0, 70, 255],
    [0, 220, 255],
    [40, 220, 80],
    [255, 235, 0],
    [255, 40, 0],
    [255, 255, 255],
  ];
  const scaled = Math.max(0, Math.min(0.9999, value)) * (stops.length - 1);
  const index = Math.floor(scaled);
  const mix = scaled - index;
  return stops[index].map((channel, component) =>
    Math.round(channel + (stops[index + 1][component] - channel) * mix));
}

function displayRange(pixels) {
  const sorted = [...pixels].sort((left, right) => left - right);
  return [sorted[Math.floor(sorted.length * 0.02)], sorted[Math.floor(sorted.length * 0.98)]];
}

function blurPixels(pixels, width, height, low, high) {
  const output = new Float32Array(pixels.length);
  const kernel = [1, 2, 1, 2, 4, 2, 1, 2, 1];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      let weight = 0;
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          const sampleX = Math.max(0, Math.min(width - 1, x + offsetX));
          const sampleY = Math.max(0, Math.min(height - 1, y + offsetY));
          const kernelWeight = kernel[(offsetY + 1) * 3 + offsetX + 1];
          const sample = Math.max(low, Math.min(high, pixels[sampleY * width + sampleX]));
          sum += sample * kernelWeight;
          weight += kernelWeight;
        }
      }
      output[y * width + x] = sum / weight;
    }
  }
  return output;
}

function resizeCanvas() {
  const scale = window.devicePixelRatio || 1;
  const width = Math.round(canvas.clientWidth * scale);
  const height = Math.round(canvas.clientHeight * scale);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function drawFrame(frame) {
  const [minimum, maximum] = displayRange(frame.pixels);
  const pixels = blurPixels(frame.pixels, frame.width, frame.height, minimum, maximum);
  const span = maximum - minimum || 1;
  const image = sensorContext.createImageData(frame.width, frame.height);
  pixels.forEach((pixel, index) => {
    const color = heatColor((pixel - minimum) / span);
    image.data.set([...color, 255], index * 4);
  });
  sensorContext.putImageData(image, 0, 0);

  resizeCanvas();
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = 'high';
  context.drawImage(sensorCanvas, 0, 0, canvas.width, canvas.height);

  const hottest = Math.max(...frame.pixels);
  const centre = frame.pixels[(frame.height / 2) * frame.width + frame.width / 2];
  readout.textContent = `${minimum.toFixed(1)}–${maximum.toFixed(1)} °C   max ${hottest.toFixed(1)} °C   ` +
    `centre ${centre.toFixed(1)} °C   sensor ${frame.ambientC.toFixed(1)} °C`;
}

function connect() {
  const socket = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`);
  socket.addEventListener('message', ({ data }) => {
    const message = JSON.parse(data);
    if (message.type !== 'frame') return;
    drawFrame(message);
  });
  socket.addEventListener('close', () => {
    setTimeout(connect, 1000);
  });
}
connect();
