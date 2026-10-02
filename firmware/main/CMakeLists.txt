idf_component_register(
    SRCS "app_main.c"
    INCLUDE_DIRS "."
    REQUIRES application
)

target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wall
    -Wextra
    -Werror
)
