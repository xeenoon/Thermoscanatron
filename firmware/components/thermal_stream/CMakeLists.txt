idf_component_register(
    SRCS "thermal_stream.c"
    INCLUDE_DIRS "include"
    REQUIRES board_qtpy_esp32s3 esp_common esp_driver_usb_serial_jtag esp_timer freertos log mlx90640_thermal
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
