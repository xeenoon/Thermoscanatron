// npm default serialport is unreliable for RTOS usage
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#define HEADER_BYTES 28U
#define PAYLOAD_WORDS 769U
#define PAYLOAD_BYTES (PAYLOAD_WORDS * 2U)
#define PACKET_BYTES (HEADER_BYTES + PAYLOAD_BYTES)

static const uint8_t magic[] = {'T', 'H', 'M', '2'};

static uint16_t read_u16_le(const uint8_t *data)
{
    return (uint16_t)data[0] | ((uint16_t)data[1] << 8U);
}

static uint32_t read_u32_le(const uint8_t *data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
}

static uint32_t crc32(const uint8_t *data, size_t length)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0; index < length; ++index) {
        crc ^= data[index];
        for (uint8_t bit = 0; bit < 8U; ++bit) {
            const uint32_t mask = (uint32_t)-(int32_t)(crc & 1U);
            crc = (crc >> 1U) ^ (0xEDB88320U & mask);
        }
    }
    return ~crc;
}

static int valid_packet(const uint8_t *packet)
{
    return packet[4] == 2U && packet[5] <= 1U &&
           read_u16_le(&packet[6]) == HEADER_BYTES &&
           read_u16_le(&packet[20]) == PAYLOAD_WORDS &&
           packet[22] == 32U && packet[23] == 24U &&
           read_u32_le(&packet[24]) == crc32(&packet[HEADER_BYTES], PAYLOAD_BYTES);
}

static size_t find_magic(const uint8_t *data, size_t length)
{
    for (size_t index = 0; index + sizeof(magic) <= length; ++index) {
        if (memcmp(&data[index], magic, sizeof(magic)) == 0) {
            return index;
        }
    }
    return length;
}

static int write_all(const uint8_t *data, size_t length)
{
    while (length > 0U) {
        const ssize_t written = write(STDOUT_FILENO, data, length);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        data += (size_t)written;
        length -= (size_t)written;
    }
    return 0;
}

static int open_serial(const char *path)
{
    const int serial = open(path, O_RDONLY | O_NOCTTY);
    if (serial < 0) {
        return -1;
    }

    struct termios options;
    if (tcgetattr(serial, &options) != 0) {
        close(serial);
        return -1;
    }
    cfmakeraw(&options);
    options.c_cflag |= CLOCAL | CREAD;
    options.c_cflag &= (tcflag_t)~HUPCL;
    options.c_cc[VMIN] = 1;
    options.c_cc[VTIME] = 0;
    (void)cfsetspeed(&options, B115200);
    if (tcsetattr(serial, TCSANOW, &options) != 0) {
        close(serial);
        return -1;
    }
    return serial;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s /dev/ttyACM0\n", argv[0]);
        return 2;
    }

    const int serial = open_serial(argv[1]);
    if (serial < 0) {
        fprintf(stderr, "%s: %s\n", argv[1], strerror(errno));
        return 1;
    }

    uint8_t buffer[PACKET_BYTES * 2U];
    size_t used = 0U;
    for (;;) {
        const ssize_t received = read(serial, &buffer[used], sizeof(buffer) - used);
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            fprintf(stderr, "serial read: %s\n", strerror(errno));
            break;
        }
        if (received == 0) {
            break;
        }
        used += (size_t)received;

        while (used >= sizeof(magic)) {
            const size_t start = find_magic(buffer, used);
            if (start == used) {
                const size_t keep = used < sizeof(magic) - 1U ? used : sizeof(magic) - 1U;
                memmove(buffer, &buffer[used - keep], keep);
                used = keep;
                break;
            }
            if (start > 0U) {
                memmove(buffer, &buffer[start], used - start);
                used -= start;
            }
            if (used < PACKET_BYTES) {
                break;
            }
            if (!valid_packet(buffer)) {
                memmove(buffer, &buffer[1], --used);
                continue;
            }
            if (write_all(buffer, PACKET_BYTES) != 0) {
                close(serial);
                return 1;
            }
            memmove(buffer, &buffer[PACKET_BYTES], used - PACKET_BYTES);
            used -= PACKET_BYTES;
        }
    }

    close(serial);
    return 1;
}
