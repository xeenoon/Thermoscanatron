#ifndef MLX90640_RAW_H
#define MLX90640_RAW_H

#include <stdint.h>

#include "esp_err.h"

#define MLX90640_RAW_WIDTH 32U
#define MLX90640_RAW_HEIGHT 24U
#define MLX90640_RAW_PIXEL_WORDS 768U
#define MLX90640_RAW_AUX_WORDS 64U
#define MLX90640_RAW_FRAME_WORDS 834U

typedef struct {
    uint16_t words[MLX90640_RAW_FRAME_WORDS];
    uint8_t subpage;
} mlx90640_raw_frame_t;

/** Initialize the sensor and set continuous chess-mode capture at 8 Hz. */
esp_err_t mlx90640_raw_init(int sda_gpio, int scl_gpio, uint32_t bus_hz);

/** Wait for and read one raw sensor subpage. */
esp_err_t mlx90640_raw_read_frame(mlx90640_raw_frame_t *frame);

/** Recover the bus after a run of failed transactions. */
esp_err_t mlx90640_raw_reset_bus(void);

#endif
