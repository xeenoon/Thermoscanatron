import { EventEmitter } from 'node:events';
import { encodePacket, FRAME_WORDS, HEIGHT, WIDTH } from './protocol.js';

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
    const words = new Uint16Array(FRAME_WORDS);
    const phase = this.#sequence / 8;
    const hotX = 15.5 + Math.cos(phase) * 8;
    const hotY = 11.5 + Math.sin(phase * 0.7) * 6;

    for (let y = 0; y < HEIGHT; y += 1) {
      for (let x = 0; x < WIDTH; x += 1) {
        const distance = Math.hypot(x - hotX, y - hotY);
        const raw = 9000 + Math.max(0, 18000 - distance * 2300) + x * 25 + y * 18;
        words[y * WIDTH + x] = Math.round(raw);
      }
    }
    for (let index = WIDTH * HEIGHT; index < FRAME_WORDS; index += 1) {
      words[index] = (0x2000 + index + this.#sequence) & 0xffff;
    }

    const packet = encodePacket({
      sequence: this.#sequence,
      timestampUs: BigInt(this.#sequence) * 125000n,
      subpage: this.#sequence & 1,
      words,
    });
    this.#sequence += 1;

    // Exercise partial serial reads, just like a real USB stream.
    for (let offset = 0; offset < packet.length; offset += 137) {
      this.emit('data', packet.subarray(offset, offset + 137));
    }
  }
}
