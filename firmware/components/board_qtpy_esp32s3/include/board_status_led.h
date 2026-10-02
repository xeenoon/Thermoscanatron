#ifndef BOARD_STATUS_LED_H
#define BOARD_STATUS_LED_H

#include <stdbool.h>

#include "esp_err.h"

/** Prepare the ADA5700 NeoPixel power and data pins. */
esp_err_t board_status_led_init(void);

/** Set the status LED to a low-brightness blue or turn it off. */
esp_err_t board_status_led_set(bool on);

#endif
