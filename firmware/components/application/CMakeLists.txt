idf_component_register(
    SRCS "application.c"
    INCLUDE_DIRS "include"
    REQUIRES board_qtpy_esp32s3 esp_common freertos log
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
