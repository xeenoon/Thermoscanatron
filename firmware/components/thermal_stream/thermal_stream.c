#include "thermal_stream.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "board_config.h"
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "mlx90640_thermal.h"

#define THERMAL_I2C_HZ 900000U
#define STREAM_VERSION 2U
#define STREAM_HEADER_BYTES 28U
/* Payload: 768 pixel temperatures then the ambient temperature, int16 centi-degrees C. */
#define STREAM_PAYLOAD_WORDS (MLX90640_THERMAL_PIXELS + 1U)
#define STREAM_PAYLOAD_BYTES (STREAM_PAYLOAD_WORDS * 2U)
#define STREAM_PACKET_BYTES (STREAM_HEADER_BYTES + STREAM_PAYLOAD_BYTES)
#define STREAM_TASK_STACK_BYTES 8192U
#define STREAM_TASK_PRIORITY 5U
#define STREAM_MAX_CONSECUTIVE_ERRORS 5U
/* Chess mode refreshes half the pixels per subpage: wait for both before sending. */
#define STREAM_WARMUP_SUBPAGES 2U

static const char *TAG = "thermal_stream";
static mlx90640_thermal_frame_t thermal_frame;
static uint8_t packet[STREAM_PACKET_BYTES];

static void put_u16_le(uint8_t *destination, uint16_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8U);
}

static void put_u32_le(uint8_t *destination, uint32_t value)
{
    for (size_t byte = 0; byte < 4U; ++byte) {
        destination[byte] = (uint8_t)(value >> (byte * 8U));
    }
}

static void put_u64_le(uint8_t *destination, uint64_t value)
{
    for (size_t byte = 0; byte < 8U; ++byte) {
        destination[byte] = (uint8_t)(value >> (byte * 8U));
    }
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

static uint16_t centi_celsius(float celsius)
{
    if (isnan(celsius)) {
        return (uint16_t)INT16_MIN;
    }
    const float centi = roundf(celsius * 100.0f);
    const float clamped = fminf(fmaxf(centi, (float)(INT16_MIN + 1)), (float)INT16_MAX);
    return (uint16_t)(int16_t)clamped;
}

static void encode_packet(uint32_t sequence)
{
    packet[0] = 'T';
    packet[1] = 'H';
    packet[2] = 'M';
    packet[3] = '2';
    packet[4] = STREAM_VERSION;
    packet[5] = thermal_frame.subpage;
    put_u16_le(&packet[6], STREAM_HEADER_BYTES);
    put_u32_le(&packet[8], sequence);
    put_u64_le(&packet[12], (uint64_t)esp_timer_get_time());
    put_u16_le(&packet[20], STREAM_PAYLOAD_WORDS);
    packet[22] = MLX90640_THERMAL_WIDTH;
    packet[23] = MLX90640_THERMAL_HEIGHT;

    for (size_t index = 0; index < MLX90640_THERMAL_PIXELS; ++index) {
        put_u16_le(
            &packet[STREAM_HEADER_BYTES + index * 2U],
            centi_celsius(thermal_frame.pixels_c[index]));
    }
    put_u16_le(
        &packet[STREAM_HEADER_BYTES + MLX90640_THERMAL_PIXELS * 2U],
        centi_celsius(thermal_frame.ambient_c));

    put_u32_le(
        &packet[24],
        crc32(&packet[STREAM_HEADER_BYTES], STREAM_PAYLOAD_BYTES));
}

static void send_packet(void)
{
    size_t sent = 0U;
    while (sent < sizeof(packet)) {
        const int written = usb_serial_jtag_write_bytes(
            &packet[sent],
            sizeof(packet) - sent,
            portMAX_DELAY);
        if (written > 0) {
            sent += (size_t)written;
        }
    }
}

static void stream_task(void *context)
{
    (void)context;

    esp_err_t error;
    do {
        error = mlx90640_thermal_init(
            BOARD_STEMMA_SDA_GPIO,
            BOARD_STEMMA_SCL_GPIO,
            THERMAL_I2C_HZ);
        if (error != ESP_OK) {
            ESP_LOGE(TAG, "MLX90640 init failed: %s", esp_err_to_name(error));
            vTaskDelay(pdMS_TO_TICKS(1000U));
        }
    } while (error != ESP_OK);

    usb_serial_jtag_driver_config_t serial_config = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    serial_config.tx_buffer_size = 8192U;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&serial_config));
    usb_serial_jtag_vfs_use_driver();
    usb_serial_jtag_vfs_set_tx_line_endings(ESP_LINE_ENDINGS_LF);

    ESP_LOGI(TAG, "MLX90640 streaming calibrated temperatures, 8 subpages/s");
    vTaskDelay(pdMS_TO_TICKS(50U));
    esp_log_level_set("*", ESP_LOG_NONE);

    uint32_t sequence = 0U;
    uint32_t consecutive_errors = 0U;
    uint32_t subpages_read = 0U;
    for (;;) {
        error = mlx90640_thermal_read(&thermal_frame);
        if (error != ESP_OK) {
            if (++consecutive_errors >= STREAM_MAX_CONSECUTIVE_ERRORS) {
                (void)mlx90640_thermal_reset_bus();
                consecutive_errors = 0U;
            }
            vTaskDelay(pdMS_TO_TICKS(1U));
            continue;
        }

        consecutive_errors = 0U;
        if (subpages_read < STREAM_WARMUP_SUBPAGES) {
            ++subpages_read;
            continue;
        }
        encode_packet(sequence++);
        send_packet();
    }
}

esp_err_t thermal_stream_start(void)
{
    const BaseType_t created = xTaskCreate(
        stream_task,
        "thermal_stream",
        STREAM_TASK_STACK_BYTES,
        NULL,
        STREAM_TASK_PRIORITY,
        NULL);

    return created == pdPASS ? ESP_OK : ESP_ERR_NO_MEM;
}
