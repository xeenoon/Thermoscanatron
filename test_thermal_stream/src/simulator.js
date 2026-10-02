import { EventEmitter } from 'node:events';
import { encodePacket, HEIGHT, PIXEL_WORDS, WIDTH } from './protocol.js';

export class SimulatedSerial extends EventEmitter {
  #timer;
  #sequence = 0;

  start() {
    this.emit('data', Buffer.from('ESP-ROM: simulated boot log\r\n'));
    this.#timer = setInterval(() => this.#emitFrame(), 125);
  }

  stop() {
    clearInterval(this.#timer);
  }

  #emitFrame() {
    const celsius = new Float32Array(PIXEL_WORDS);
    const phase = this.#sequence / 8;
    const hotX = 15.5 + Math.cos(phase) * 8;
    const hotY = 11.5 + Math.sin(phase * 0.7) * 6;

    for (let y = 0; y < HEIGHT; y += 1) {
      for (let x = 0; x < WIDTH; x += 1) {
        const distance = Math.hypot(x - hotX, y - hotY);
        // Room at ~22 C with a ~34 C hand-sized warm blob drifting around.
        celsius[y * WIDTH + x] = 22 + Math.max(0, 12 - distance * 1.6) + x * 0.02 + y * 0.015;
      }
    }

    const packet = encodePacket({
      sequence: this.#sequence,
      timestampUs: BigInt(this.#sequence) * 125000n,
      subpage: this.#sequence & 1,
      celsius,
      ambientC: 27.5,
    });
    this.#sequence += 1;

    // Exercise partial serial reads, just like a real USB stream.
    for (let offset = 0; offset < packet.length; offset += 137) {
      this.emit('data', packet.subarray(offset, offset + 137));
    }
  }
}
