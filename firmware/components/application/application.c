#include "application.h"

#include "board_status_led.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "thermal_stream.h"

#define HEARTBEAT_PERIOD_MS 1000U
#define HEARTBEAT_TASK_STACK_BYTES 3072U
#define HEARTBEAT_TASK_PRIORITY 2U

static const char *TAG = "application";

static void heartbeat_task(void *context)
{
    (void)context;

    const TickType_t period = pdMS_TO_TICKS(HEARTBEAT_PERIOD_MS);
    TickType_t next_wake = xTaskGetTickCount();

    for (;;) {
        ESP_ERROR_CHECK(board_status_led_set(true));
        vTaskDelay(pdMS_TO_TICKS(100U));
        ESP_ERROR_CHECK(board_status_led_set(false));

        vTaskDelayUntil(&next_wake, period);
    }
}

esp_err_t application_start(void)
{
    esp_err_t error = board_status_led_init();
    if (error != ESP_OK) {
        return error;
    }

    ESP_LOGI(TAG, "Hello, world!");

    const BaseType_t created = xTaskCreate(
        heartbeat_task,
        "heartbeat",
        HEARTBEAT_TASK_STACK_BYTES,
        NULL,
        HEARTBEAT_TASK_PRIORITY,
        NULL);

    if (created != pdPASS) {
        return ESP_ERR_NO_MEM;
    }

    return thermal_stream_start();
}
