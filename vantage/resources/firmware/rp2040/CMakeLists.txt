# RP2040 trigger box (Vantage). Build: see README.md.
cmake_minimum_required(VERSION 3.13)
include(pico_sdk_import.cmake)
project(vantage_trigger C CXX ASM)
set(CMAKE_C_STANDARD 11)
pico_sdk_init()

add_executable(vantage_trigger main.c trigger_core.c)
pico_generate_pio_header(vantage_trigger ${CMAKE_CURRENT_LIST_DIR}/trigger.pio)
target_include_directories(vantage_trigger PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(vantage_trigger pico_stdlib hardware_pio hardware_clocks hardware_watchdog)
pico_enable_stdio_usb(vantage_trigger 1)
pico_enable_stdio_uart(vantage_trigger 0)
pico_add_extra_outputs(vantage_trigger)
