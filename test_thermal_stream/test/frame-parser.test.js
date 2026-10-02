import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameParser } from '../src/frame-parser.js';
import { encodePacket, PIXEL_WORDS } from '../src/protocol.js';

function fixture(sequence = 42) {
  const celsius = Float32Array.from({ length: PIXEL_WORDS }, (_, index) => 20 + index / 100);
  return encodePacket({ sequence, timestampUs: 123456789n, subpage: sequence & 1, celsius, ambientC: 26.25 });
}

test('parses a firmware-format packet split across arbitrary serial chunks', () => {
  const frames = [];
  const parser = new FrameParser({ onFrame: (frame) => frames.push(frame) });
  const input = Buffer.concat([Buffer.from('boot noise\r\n'), fixture()]);
  for (let offset = 0; offset < input.length; offset += 31) parser.push(input.subarray(offset, offset + 31));

  assert.equal(frames.length, 1);
  assert.equal(frames[0].sequence, 42);
  assert.equal(frames[0].timestampUs, 123456789n);
  assert.ok(Math.abs(frames[0].celsius[767] - 27.67) < 1e-4);
  assert.equal(frames[0].ambientC, 26.25);
});

test('drops a corrupt packet and resynchronizes at the following frame', () => {
  const frames = [];
  const errors = [];
  const bad = fixture(4);
  bad[100] ^= 0xff;
  const parser = new FrameParser({
    onFrame: (frame) => frames.push(frame),
    onError: (error) => errors.push(error),
  });
  parser.push(Buffer.concat([bad, fixture(5)]));

  assert.equal(errors.length, 1);
  assert.deepEqual(frames.map((frame) => frame.sequence), [5]);
});

test('round-trips negative temperatures', () => {
  const frames = [];
  const parser = new FrameParser({ onFrame: (frame) => frames.push(frame) });
  const celsius = new Float32Array(PIXEL_WORDS).fill(-12.34);
  parser.push(encodePacket({ sequence: 1, timestampUs: 0n, subpage: 0, celsius, ambientC: -5 }));

  assert.ok(Math.abs(frames[0].celsius[0] + 12.34) < 1e-4);
  assert.equal(frames[0].ambientC, -5);
});
