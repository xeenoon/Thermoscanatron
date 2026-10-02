export const MAGIC = Buffer.from('THM2');
export const VERSION = 2;
export const HEADER_BYTES = 28;
export const WIDTH = 32;
export const HEIGHT = 24;
export const PIXEL_WORDS = WIDTH * HEIGHT;
// 768 pixel temperatures followed by the sensor ambient temperature, int16 centi-degrees C.
export const PAYLOAD_WORDS = PIXEL_WORDS + 1;
export const PAYLOAD_BYTES = PAYLOAD_WORDS * 2;
export const PACKET_BYTES = HEADER_BYTES + PAYLOAD_BYTES;

export function crc32(data) {
  let crc = 0xffffffff;
  for (const byte of data) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

export function decodePacket(packet) {
  if (packet.length !== PACKET_BYTES || !packet.subarray(0, 4).equals(MAGIC)) {
    throw new Error('invalid packet boundary');
  }

  const version = packet.readUInt8(4);
  const subpage = packet.readUInt8(5);
  const headerBytes = packet.readUInt16LE(6);
  const sequence = packet.readUInt32LE(8);
  const timestampUs = packet.readBigUInt64LE(12);
  const wordCount = packet.readUInt16LE(20);
  const width = packet.readUInt8(22);
  const height = packet.readUInt8(23);
  const expectedCrc = packet.readUInt32LE(24);

  if (version !== VERSION || headerBytes !== HEADER_BYTES || wordCount !== PAYLOAD_WORDS) {
    throw new Error('unsupported packet format');
  }
  if (width !== WIDTH || height !== HEIGHT || subpage > 1) {
    throw new Error('invalid frame metadata');
  }

  const payload = packet.subarray(headerBytes);
  const actualCrc = crc32(payload);
  if (actualCrc !== expectedCrc) {
    throw new Error('payload CRC mismatch');
  }

  const celsius = new Float32Array(PIXEL_WORDS);
  for (let index = 0; index < PIXEL_WORDS; index += 1) {
    celsius[index] = payload.readInt16LE(index * 2) / 100;
  }
  const ambientC = payload.readInt16LE(PIXEL_WORDS * 2) / 100;

  return { version, subpage, sequence, timestampUs, width, height, celsius, ambientC };
}

export function encodePacket({ sequence, timestampUs, subpage, celsius, ambientC }) {
  if (celsius.length !== PIXEL_WORDS) throw new Error(`expected ${PIXEL_WORDS} pixels`);

  const packet = Buffer.alloc(PACKET_BYTES);
  MAGIC.copy(packet);
  packet.writeUInt8(VERSION, 4);
  packet.writeUInt8(subpage, 5);
  packet.writeUInt16LE(HEADER_BYTES, 6);
  packet.writeUInt32LE(sequence >>> 0, 8);
  packet.writeBigUInt64LE(BigInt(timestampUs), 12);
  packet.writeUInt16LE(PAYLOAD_WORDS, 20);
  packet.writeUInt8(WIDTH, 22);
  packet.writeUInt8(HEIGHT, 23);
  for (let index = 0; index < PIXEL_WORDS; index += 1) {
    packet.writeInt16LE(Math.round(celsius[index] * 100), HEADER_BYTES + index * 2);
  }
  packet.writeInt16LE(Math.round(ambientC * 100), HEADER_BYTES + PIXEL_WORDS * 2);
  packet.writeUInt32LE(crc32(packet.subarray(HEADER_BYTES)), 24);
  return packet;
}
