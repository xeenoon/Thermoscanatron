#ifndef MLX90640_THERMAL_H
#define MLX90640_THERMAL_H

#include <stdint.h>

#include "esp_err.h"

#define MLX90640_THERMAL_WIDTH 32U
#define MLX90640_THERMAL_HEIGHT 24U
#define MLX90640_THERMAL_PIXELS 768U

typedef struct {
    /** Object temperatures in degrees C, row-major. Each read refreshes one chess-pattern
     *  subpage (half the pixels); the other half keeps its value from the previous read. */
    float pixels_c[MLX90640_THERMAL_PIXELS];
    /** Sensor die (ambient) temperature in degrees C. */
    float ambient_c;
    uint8_t subpage;
} mlx90640_thermal_frame_t;

/**
 * Initialize the sensor: read its EEPROM calibration and set continuous chess-mode capture
 * at 8 subpages per second.
 */
esp_err_t mlx90640_thermal_init(int sda_gpio, int scl_gpio, uint32_t bus_hz);

/** Wait for the next subpage and convert it to calibrated temperatures in place. */
esp_err_t mlx90640_thermal_read(mlx90640_thermal_frame_t *frame);

/** Recover the bus after a run of failed transactions. */
esp_err_t mlx90640_thermal_reset_bus(void);

#endif
