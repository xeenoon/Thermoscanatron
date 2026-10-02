cmake_minimum_required(VERSION 3.20)
project(thermal_serial C)

add_executable(thermal_serial thermal_serial.c)
target_compile_features(thermal_serial PRIVATE c_std_11)
target_compile_definitions(thermal_serial PRIVATE _DEFAULT_SOURCE)
target_compile_options(thermal_serial PRIVATE -Wall -Wextra -Wpedantic -Werror)
