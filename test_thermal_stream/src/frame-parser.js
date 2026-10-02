import { decodePacket, MAGIC, PACKET_BYTES } from './protocol.js';

const MAX_BUFFER_BYTES = PACKET_BYTES * 3;

export class FrameParser {
  #buffer = Buffer.alloc(0);

  constructor({ onFrame, onError = () => {} }) {
    this.onFrame = onFrame;
    this.onError = onError;
  }

  push(chunk) {
    this.#buffer = Buffer.concat([this.#buffer, chunk]);

    while (this.#buffer.length >= MAGIC.length) {
      const magicOffset = this.#buffer.indexOf(MAGIC);
      if (magicOffset < 0) {
        this.#buffer = this.#buffer.subarray(Math.max(0, this.#buffer.length - 3));
        return;
      }
      if (magicOffset > 0) this.#buffer = this.#buffer.subarray(magicOffset);
      if (this.#buffer.length < PACKET_BYTES) return;

      const candidate = this.#buffer.subarray(0, PACKET_BYTES);
      try {
        this.onFrame(decodePacket(candidate));
        this.#buffer = this.#buffer.subarray(PACKET_BYTES);
      } catch (error) {
        this.onError(error);
        this.#buffer = this.#buffer.subarray(1);
      }
    }

    if (this.#buffer.length > MAX_BUFFER_BYTES) {
      this.#buffer = this.#buffer.subarray(-3);
    }
  }
}
