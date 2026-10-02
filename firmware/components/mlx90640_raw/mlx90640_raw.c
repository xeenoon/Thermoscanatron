#include "mlx90640_raw.h"

#include <stddef.h>

#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MLX90640_ADDRESS 0x33U
#define MLX90640_STATUS_REGISTER 0x8000U
#define MLX90640_CONTROL_REGISTER 0x800DU
#define MLX90640_PIXEL_START_REGISTER 0x0400U
#define MLX90640_AUX_START_REGISTER 0x0700U
#define MLX90640_DATA_READY_MASK 0x0008U
#define MLX90640_SUBPAGE_MASK 0x0001U
#define MLX90640_STATUS_CLEAR_VALUE 0x0030U
#define MLX90640_REFRESH_RATE_MASK 0x0380U
#define MLX90640_REFRESH_RATE_8_HZ (4U << 7U)
#define MLX90640_CHESS_MODE_MASK 0x1000U
#define MLX90640_TRANSACTION_TIMEOUT_MS 100
#define MLX90640_FRAME_TIMEOUT_MS 500U

static i2c_master_bus_handle_t sensor_bus;
static i2c_master_dev_handle_t sensor_device;
static uint8_t transfer_buffer[MLX90640_RAW_PIXEL_WORDS * 2U];

static void release_bus(void)
{
    if (sensor_device != NULL) {
        (void)i2c_master_bus_rm_device(sensor_device);
        sensor_device = NULL;
    }
    if (sensor_bus != NULL) {
        (void)i2c_del_master_bus(sensor_bus);
        sensor_bus = NULL;
    }
}

static esp_err_t read_words(uint16_t start_register, uint16_t *words, size_t word_count)
{
    if (word_count > MLX90640_RAW_PIXEL_WORDS) {
        return ESP_ERR_INVALID_SIZE;
    }

    const uint8_t address[2] = {
        (uint8_t)(start_register >> 8U),
        (uint8_t)start_register,
    };
    const size_t byte_count = word_count * 2U;
    esp_err_t error = i2c_master_transmit_receive(
        sensor_device,
        address,
        sizeof(address),
        transfer_buffer,
        byte_count,
        MLX90640_TRANSACTION_TIMEOUT_MS);
    if (error != ESP_OK) {
        return error;
    }

    for (size_t index = 0; index < word_count; ++index) {
        words[index] = ((uint16_t)transfer_buffer[index * 2U] << 8U) |
                       transfer_buffer[index * 2U + 1U];
    }

    return ESP_OK;
}

static esp_err_t write_word(uint16_t target_register, uint16_t value)
{
    const uint8_t transaction[4] = {
        (uint8_t)(target_register >> 8U),
        (uint8_t)target_register,
        (uint8_t)(value >> 8U),
        (uint8_t)value,
    };

    return i2c_master_transmit(
        sensor_device,
        transaction,
        sizeof(transaction),
        MLX90640_TRANSACTION_TIMEOUT_MS);
}

esp_err_t mlx90640_raw_init(int sda_gpio, int scl_gpio, uint32_t bus_hz)
{
    if (sensor_device != NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    const i2c_master_bus_config_t bus_config = {
        .i2c_port = -1,
        .sda_io_num = sda_gpio,
        .scl_io_num = scl_gpio,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7U,
        .flags.enable_internal_pullup = true,
    };
    esp_err_t error = i2c_new_master_bus(&bus_config, &sensor_bus);
    if (error != ESP_OK) {
        return error;
    }

    error = i2c_master_probe(sensor_bus, MLX90640_ADDRESS, MLX90640_TRANSACTION_TIMEOUT_MS);
    if (error != ESP_OK) {
        release_bus();
        return error;
    }

    const i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MLX90640_ADDRESS,
        .scl_speed_hz = bus_hz,
        .scl_wait_us = 0U,
    };
    error = i2c_master_bus_add_device(sensor_bus, &device_config, &sensor_device);
    if (error != ESP_OK) {
        release_bus();
        return error;
    }

    uint16_t control_register;
    error = read_words(MLX90640_CONTROL_REGISTER, &control_register, 1U);
    if (error != ESP_OK) {
        release_bus();
        return error;
    }

    control_register &= (uint16_t)~MLX90640_REFRESH_RATE_MASK;
    control_register |= MLX90640_REFRESH_RATE_8_HZ | MLX90640_CHESS_MODE_MASK;
    error = write_word(MLX90640_CONTROL_REGISTER, control_register);
    if (error != ESP_OK) {
        release_bus();
    }
    return error;
}

esp_err_t mlx90640_raw_read_frame(mlx90640_raw_frame_t *frame)
{
    if (sensor_device == NULL || frame == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    uint16_t status_register = 0U;
    const TickType_t started = xTaskGetTickCount();
    const TickType_t timeout = pdMS_TO_TICKS(MLX90640_FRAME_TIMEOUT_MS);

    do {
        esp_err_t error = read_words(MLX90640_STATUS_REGISTER, &status_register, 1U);
        if (error != ESP_OK) {
            return error;
        }
        if ((status_register & MLX90640_DATA_READY_MASK) == 0U) {
            vTaskDelay(pdMS_TO_TICKS(1U));
        }
    } while ((status_register & MLX90640_DATA_READY_MASK) == 0U &&
             (xTaskGetTickCount() - started) < timeout);

    if ((status_register & MLX90640_DATA_READY_MASK) == 0U) {
        return ESP_ERR_TIMEOUT;
    }

    frame->subpage = (uint8_t)(status_register & MLX90640_SUBPAGE_MASK);

    esp_err_t error = write_word(MLX90640_STATUS_REGISTER, MLX90640_STATUS_CLEAR_VALUE);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(
        MLX90640_PIXEL_START_REGISTER,
        frame->words,
        MLX90640_RAW_PIXEL_WORDS);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(
        MLX90640_AUX_START_REGISTER,
        &frame->words[MLX90640_RAW_PIXEL_WORDS],
        MLX90640_RAW_AUX_WORDS);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(
        MLX90640_CONTROL_REGISTER,
        &frame->words[MLX90640_RAW_PIXEL_WORDS + MLX90640_RAW_AUX_WORDS],
        1U);
    if (error != ESP_OK) {
        return error;
    }

    frame->words[MLX90640_RAW_FRAME_WORDS - 1U] = frame->subpage;
    return ESP_OK;
}

esp_err_t mlx90640_raw_reset_bus(void)
{
    return sensor_bus == NULL ? ESP_ERR_INVALID_STATE : i2c_master_bus_reset(sensor_bus);
}
