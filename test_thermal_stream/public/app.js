const canvas = document.querySelector('#thermal');
const context = canvas.getContext('2d');

function heatColor(value) {
  const red = Math.round(Math.max(0, Math.min(1, value)) * 255);
  return [red, 0, 255 - red];
}

function drawFrame(frame) {
  const minimum = Math.min(...frame.pixels);
  const maximum = Math.max(...frame.pixels);
  const span = Math.max(1, maximum - minimum);
  const image = context.createImageData(frame.width, frame.height);
  frame.pixels.forEach((pixel, index) => {
    const color = heatColor((pixel - minimum) / span);
    image.data.set([...color, 255], index * 4);
  });
  context.putImageData(image, 0, 0);
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
