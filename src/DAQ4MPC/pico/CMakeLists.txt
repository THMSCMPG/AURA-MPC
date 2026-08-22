# pico/CMakeLists.txt – pico-sdk build configuration for AURA-MFP Pico firmware.
#
# Build instructions (from the pico/ directory or a build/ sub-directory):
#
#   cmake -DPICO_SDK_PATH=/path/to/pico-sdk -B build
#   cmake --build build -j4
#
# Flash the resulting UF2 image:
#
#   picotool load build/aura_edge_pico.uf2
#
# Or drag-and-drop the UF2 onto the RP2040 mass-storage device while holding
# the BOOTSEL button.

cmake_minimum_required(VERSION 3.13)

# Pull in the pico-sdk (must be set via -DPICO_SDK_PATH or the environment
# variable PICO_SDK_PATH before the project() call).
if(NOT DEFINED PICO_SDK_PATH)
    set(PICO_SDK_PATH $ENV{PICO_SDK_PATH})
endif()
if(NOT EXISTS "${PICO_SDK_PATH}/external/pico_sdk_import.cmake")
    message(FATAL_ERROR
        "pico-sdk not found at '${PICO_SDK_PATH}'.  Set PICO_SDK_PATH to the "
        "pico-sdk root directory, e.g.:  cmake -DPICO_SDK_PATH=/opt/pico-sdk ..")
endif()
include(${PICO_SDK_PATH}/external/pico_sdk_import.cmake)

project(aura_edge_pico C CXX ASM)
set(CMAKE_C_STANDARD 11)

pico_sdk_init()

# ── Firmware target ────────────────────────────────────────────────────────
add_executable(aura_edge_pico
    main.c
    adc_sensors.c
    rtc_ds3231.c
    json_builder.c
    uart_output.c
)

target_include_directories(aura_edge_pico PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(aura_edge_pico
    pico_stdlib
    hardware_adc
    hardware_i2c
    hardware_uart
    hardware_gpio
    pico_stdio_usb
)

# Enable USB CDC for development/debug mirror output.
# UART stdio is disabled so that UART0 is available exclusively for the
# JSON stream (managed by uart_output.c).
pico_enable_stdio_usb(aura_edge_pico 1)
pico_enable_stdio_uart(aura_edge_pico 0)

# Generate UF2, map, and disassembly alongside the ELF.
pico_add_extra_outputs(aura_edge_pico)
