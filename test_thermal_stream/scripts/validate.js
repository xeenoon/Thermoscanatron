import assert from 'node:assert/strict';
import { FrameParser } from '../src/frame-parser.js';
import { SimulatedSerial } from '../src/simulator.js';

let frames = 0;
let firstSequence;
let lastSequence;
let parseErrors = 0;
const parser = new FrameParser({
  onFrame(frame) {
    frames += 1;
    firstSequence ??= frame.sequence;
    lastSequence = frame.sequence;
    assert.equal(frame.words.length, 834);
    assert.equal(frame.width * frame.height, 768);
  },
  onError() { parseErrors += 1; },
});
const source = new SimulatedSerial();
source.on('data', (chunk) => parser.push(chunk));
source.start();

setTimeout(() => {
  source.stop();
  assert.ok(frames >= 7, `expected at least 7 frames, received ${frames}`);
  assert.equal(lastSequence - firstSequence, frames - 1);
  assert.equal(parseErrors, 0);
  console.log(`Validated ${frames} sequential THM1 frames: 32x24 pixels, 834 words, CRC clean.`);
}, 1100);
