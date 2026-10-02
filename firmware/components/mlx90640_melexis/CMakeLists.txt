# Melexis MLX90640 reference driver, vendored unmodified (Apache-2.0, see LICENSE) from
# https://github.com/melexis/mlx90640-library at f6be7ca. The I2C callbacks declared in
# MLX90640_I2C_Driver.h are implemented by mlx90640_thermal.
idf_component_register(
    SRCS "MLX90640_API.c"
    INCLUDE_DIRS "include"
)

# Upstream code; keep its warnings out of the build and optimise the per-frame maths even in
# debug builds.
target_compile_options(${COMPONENT_LIB} PRIVATE -w -O2)
