idf_component_register(
    SRCS "mlx90640_raw.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_common esp_driver_i2c freertos
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
