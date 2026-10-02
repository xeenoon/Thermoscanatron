import assert from 'node:assert/strict';
import test from 'node:test';
import { FrameParser } from '../src/frame-parser.js';
import { encodePacket, FRAME_WORDS, signedWord } from '../src/protocol.js';

function fixture(sequence = 42) {
  const words = Uint16Array.from({ length: FRAME_WORDS }, (_, index) => index * 17 & 0xffff);
  return encodePacket({ sequence, timestampUs: 123456789n, subpage: sequence & 1, words });
}

test('parses a firmware-format packet split across arbitrary serial chunks', () => {
  const frames = [];
  const parser = new FrameParser({ onFrame: (frame) => frames.push(frame) });
  const input = Buffer.concat([Buffer.from('boot noise\r\n'), fixture()]);
  for (let offset = 0; offset < input.length; offset += 31) parser.push(input.subarray(offset, offset + 31));

  assert.equal(frames.length, 1);
  assert.equal(frames[0].sequence, 42);
  assert.equal(frames[0].timestampUs, 123456789n);
  assert.equal(frames[0].words[767], 767 * 17);
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

test('converts raw MLX90640 words to signed pixel readings', () => {
  assert.equal(signedWord(0x0001), 1);
  assert.equal(signedWord(0x7fff), 32767);
  assert.equal(signedWord(0xffff), -1);
  assert.equal(signedWord(0x8000), -32768);
});
