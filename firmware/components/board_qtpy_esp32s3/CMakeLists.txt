idf_component_register(
    SRCS "board_status_led.c"
    INCLUDE_DIRS "include"
    REQUIRES driver esp_common led_strip
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
