#ifndef THERMAL_STREAM_H
#define THERMAL_STREAM_H

#include "esp_err.h"

/** Start calibrated MLX90640 capture and USB serial transmission. */
esp_err_t thermal_stream_start(void);

#endif
