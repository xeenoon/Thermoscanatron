#include "mlx90640_thermal.h"

#include <stddef.h>

#include "MLX90640_API.h"
#include "MLX90640_I2C_Driver.h"
#include "driver/i2c_master.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define MLX90640_ADDRESS 0x33U
#define MLX90640_STATUS_REGISTER 0x8000U
#define MLX90640_CONTROL_REGISTER 0x800DU
#define MLX90640_PIXEL_START_REGISTER 0x0400U
#define MLX90640_AUX_START_REGISTER 0x0700U
#define MLX90640_AUX_WORDS 64U
#define MLX90640_FRAME_WORDS 834U
#define MLX90640_AUX_INVALID 0x7FFFU
#define MLX90640_DATA_READY_MASK 0x0008U
#define MLX90640_SUBPAGE_MASK 0x0001U
#define MLX90640_STATUS_CLEAR_VALUE 0x0030U
#define MLX90640_REFRESH_RATE_MASK 0x0380U
#define MLX90640_REFRESH_RATE_8_HZ (4U << 7U)
#define MLX90640_CHESS_MODE_MASK 0x1000U
#define MLX90640_TRANSACTION_TIMEOUT_MS 100
#define MLX90640_FRAME_TIMEOUT_MS 500U
#define MLX90640_CHESS_MODE 1
#define MLX90640_EMISSIVITY 0.95f
/* Melexis: in open air the reflected temperature is the sensor ambient minus 8 degrees C. */
#define MLX90640_TA_SHIFT_C 8.0f

static i2c_master_bus_handle_t sensor_bus;
static i2c_master_dev_handle_t sensor_device;
static uint8_t transfer_buffer[MLX90640_THERMAL_PIXELS * 2U];
static uint16_t eeprom[MLX90640_EEPROM_DUMP_NUM];
static paramsMLX90640 calibration;
/* Melexis frame layout: 768 pixel words, 64 aux words, control register, subpage. */
static uint16_t frame_words[MLX90640_FRAME_WORDS];

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
    if (word_count > MLX90640_THERMAL_PIXELS) {
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

/* I2C callbacks required by the Melexis library (MLX90640_I2C_Driver.h). */
void MLX90640_I2CInit(void)
{
}

void MLX90640_I2CFreqSet(int freq)
{
    (void)freq;
}

int MLX90640_I2CGeneralReset(void)
{
    return mlx90640_thermal_reset_bus() == ESP_OK ? 0 : -1;
}

int MLX90640_I2CRead(uint8_t slaveAddr, uint16_t startAddress, uint16_t nMemAddressRead, uint16_t *data)
{
    (void)slaveAddr;
    while (nMemAddressRead > 0U) {
        const uint16_t count = nMemAddressRead > MLX90640_THERMAL_PIXELS
            ? (uint16_t)MLX90640_THERMAL_PIXELS
            : nMemAddressRead;
        if (read_words(startAddress, data, count) != ESP_OK) {
            return -1;
        }
        startAddress += count;
        data += count;
        nMemAddressRead -= count;
    }
    return 0;
}

int MLX90640_I2CWrite(uint8_t slaveAddr, uint16_t writeAddress, uint16_t data)
{
    (void)slaveAddr;
    return write_word(writeAddress, data) == ESP_OK ? 0 : -1;
}

esp_err_t mlx90640_thermal_init(int sda_gpio, int scl_gpio, uint32_t bus_hz)
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

    if (MLX90640_DumpEE(MLX90640_ADDRESS, eeprom) != 0) {
        release_bus();
        return ESP_FAIL;
    }
    /* Bad- and outlier-pixel warnings are fine (they are corrected per frame); only a
     * corrupt EEPROM is fatal. */
    if (MLX90640_ExtractParameters(eeprom, &calibration) == -MLX90640_EEPROM_DATA_ERROR) {
        release_bus();
        return ESP_ERR_INVALID_RESPONSE;
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

esp_err_t mlx90640_thermal_read(mlx90640_thermal_frame_t *frame)
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

    const uint8_t subpage = (uint8_t)(status_register & MLX90640_SUBPAGE_MASK);

    esp_err_t error = write_word(MLX90640_STATUS_REGISTER, MLX90640_STATUS_CLEAR_VALUE);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(MLX90640_PIXEL_START_REGISTER, frame_words, MLX90640_THERMAL_PIXELS);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(
        MLX90640_AUX_START_REGISTER,
        &frame_words[MLX90640_THERMAL_PIXELS],
        MLX90640_AUX_WORDS);
    if (error != ESP_OK) {
        return error;
    }

    error = read_words(
        MLX90640_CONTROL_REGISTER,
        &frame_words[MLX90640_THERMAL_PIXELS + MLX90640_AUX_WORDS],
        1U);
    if (error != ESP_OK) {
        return error;
    }
    frame_words[MLX90640_FRAME_WORDS - 1U] = subpage;

    /* A glitched read returns 0x7FFF in the supply and ambient aux words. */
    if (frame_words[MLX90640_THERMAL_PIXELS] == MLX90640_AUX_INVALID ||
        frame_words[MLX90640_THERMAL_PIXELS + 32U] == MLX90640_AUX_INVALID) {
        return ESP_ERR_INVALID_RESPONSE;
    }

    const float ambient_c = MLX90640_GetTa(frame_words, &calibration);
    MLX90640_CalculateTo(
        frame_words,
        &calibration,
        MLX90640_EMISSIVITY,
        ambient_c - MLX90640_TA_SHIFT_C,
        frame->pixels_c);
    MLX90640_BadPixelsCorrection(calibration.brokenPixels, frame->pixels_c, MLX90640_CHESS_MODE, &calibration);
    MLX90640_BadPixelsCorrection(calibration.outlierPixels, frame->pixels_c, MLX90640_CHESS_MODE, &calibration);

    frame->ambient_c = ambient_c;
    frame->subpage = subpage;
    return ESP_OK;
}

esp_err_t mlx90640_thermal_reset_bus(void)
{
    return sensor_bus == NULL ? ESP_ERR_INVALID_STATE : i2c_master_bus_reset(sensor_bus);
}
