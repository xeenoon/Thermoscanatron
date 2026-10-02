#include "board_status_led.h"

#include "driver/gpio.h"
#include "led_strip.h"

#define QTPY_NEOPIXEL_DATA_GPIO GPIO_NUM_39
#define QTPY_NEOPIXEL_POWER_GPIO GPIO_NUM_38
#define QTPY_NEOPIXEL_COUNT 1U
#define STATUS_BLUE_LEVEL 16U

static led_strip_handle_t status_pixel;

esp_err_t board_status_led_init(void)
{
    const gpio_config_t power_config = {
        .pin_bit_mask = 1ULL << QTPY_NEOPIXEL_POWER_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };

    esp_err_t error = gpio_config(&power_config);
    if (error != ESP_OK) {
        return error;
    }

    error = gpio_set_level(QTPY_NEOPIXEL_POWER_GPIO, 1);
    if (error != ESP_OK) {
        return error;
    }

    const led_strip_config_t strip_config = {
        .strip_gpio_num = QTPY_NEOPIXEL_DATA_GPIO,
        .max_leds = QTPY_NEOPIXEL_COUNT,
        .led_model = LED_MODEL_WS2812,
        .color_component_format = LED_STRIP_COLOR_COMPONENT_FMT_GRB,
        .flags.invert_out = false,
    };
    const led_strip_rmt_config_t rmt_config = {
        .clk_src = RMT_CLK_SRC_DEFAULT,
        .resolution_hz = 10U * 1000U * 1000U,
        .mem_block_symbols = 0U,
        .flags.with_dma = false,
    };

    error = led_strip_new_rmt_device(&strip_config, &rmt_config, &status_pixel);
    if (error != ESP_OK) {
        return error;
    }

    return led_strip_clear(status_pixel);
}

esp_err_t board_status_led_set(bool on)
{
    if (status_pixel == NULL) {
        return ESP_ERR_INVALID_STATE;
    }

    const uint8_t blue = on ? STATUS_BLUE_LEVEL : 0U;
    esp_err_t error = led_strip_set_pixel(status_pixel, 0U, 0U, 0U, blue);
    if (error != ESP_OK) {
        return error;
    }

    return led_strip_refresh(status_pixel);
}
