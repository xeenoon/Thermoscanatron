import { createReadStream, readdirSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { WebSocketServer } from 'ws';
import { FrameParser } from './frame-parser.js';
import { PIXEL_WORDS, signedWord } from './protocol.js';
import { SimulatedSerial } from './simulator.js';

const args = process.argv.slice(2);
const valueAfter = (name) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : undefined;
};

if (args.includes('--list')) {
  const ports = readdirSync('/dev')
    .filter((name) => /^tty(?:ACM|USB)\d+$/.test(name))
    .map((name) => `/dev/${name}`);
  if (ports.length === 0) console.log('No serial ports found.');
  for (const port of ports) console.log(port);
  process.exit(0);
}

const simulate = args.includes('--simulate');
const serialPath = valueAfter('--serial') ?? '/dev/ttyACM0';
const httpPort = Number(valueAfter('--http-port') ?? process.env.PORT ?? 3000);

const publicDirectory = fileURLToPath(new URL('../public', import.meta.url));
const contentTypes = { '.css': 'text/css', '.html': 'text/html', '.js': 'text/javascript' };
const server = createServer((request, response) => {
  const requested = request.url === '/' ? '/index.html' : request.url;
  if (!['/index.html', '/app.js', '/style.css'].includes(requested)) {
    response.writeHead(404).end('Not found');
    return;
  }
  response.setHeader('Content-Type', contentTypes[extname(requested)]);
  createReadStream(join(publicDirectory, requested)).pipe(response);
});

const sockets = new WebSocketServer({ server });
let frames = 0;
let parseErrors = 0;
let latestMessage;
const parser = new FrameParser({
  onFrame(frame) {
    frames += 1;
    latestMessage = JSON.stringify({
      type: 'frame',
      sequence: frame.sequence,
      timestampUs: frame.timestampUs.toString(),
      subpage: frame.subpage,
      width: frame.width,
      height: frame.height,
      pixels: Array.from(frame.words.subarray(0, PIXEL_WORDS), signedWord),
      frames,
      parseErrors,
    });
    for (const socket of sockets.clients) {
      if (socket.readyState === socket.OPEN) socket.send(latestMessage);
    }
  },
  onError(error) {
    parseErrors += 1;
    console.warn(`Dropped packet: ${error.message}`);
  },
});
sockets.on('connection', (socket) => {
  if (latestMessage) socket.send(latestMessage);
});

const nativeReader = fileURLToPath(new URL('../native/build/thermal_serial', import.meta.url));
let child;
let stopping = false;

function startNativeReader() {
  child = spawn(nativeReader, [serialPath], { stdio: ['ignore', 'pipe', 'inherit'] });
  child.stdout.on('data', (chunk) => parser.push(chunk));
  child.stdout.on('error', (error) => console.error(`Stream error: ${error.message}`));
  child.on('error', (error) => console.error(`Native reader error: ${error.message}`));
  child.on('exit', () => {
    if (!stopping) setTimeout(startNativeReader, 250);
  });
}

let simulator;
if (simulate) {
  simulator = new SimulatedSerial();
  simulator.on('data', (chunk) => parser.push(chunk));
  simulator.start();
} else {
  startNativeReader();
}

server.listen(httpPort, () => {
  console.log(`Thermal viewer: http://localhost:${httpPort}`);
  console.log(simulate ? 'Input: simulated THM1 stream' : `Input: ${serialPath}`);
});

function shutdown() {
  stopping = true;
  simulator?.stop();
  child?.kill('SIGTERM');
  sockets.close();
  server.close(() => process.exit(0));
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
