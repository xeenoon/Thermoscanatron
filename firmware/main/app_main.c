#include "application.h"

#include "esp_err.h"
#include "esp_log.h"

static const char *TAG = "main";

void app_main(void)
{
    ESP_ERROR_CHECK(application_start());
    ESP_LOGI(TAG, "Application started");
}
