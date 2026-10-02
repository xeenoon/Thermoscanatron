idf_component_register(
    SRCS "mlx90640_thermal.c"
    INCLUDE_DIRS "include"
    REQUIRES esp_common esp_driver_i2c freertos
    PRIV_REQUIRES mlx90640_melexis
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
