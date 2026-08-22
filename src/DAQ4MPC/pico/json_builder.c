/* pico/json_builder.c -- Build the Pico's RAW packet JSON strings.
 *
 * See json_builder.h for the full PICO_RAW_PACKET_SCHEMA rationale.
 * No calibration, no unit conversion, no null-handling needed here --
 * raw ADC counts are always present integers, not optionally-absent
 * physical readings.
 */

#include "json_builder.h"

#include <stdio.h>

int json_build_packet(char *buf, size_t buf_len,
                      const char *timestamp_str,
                      float t_s,
                      uint16_t t_amb_raw, uint16_t ws_raw,
                      float lat,   float lon,
                      uint16_t fault_flags,
                      const char *edge_version)
{
    if (!buf || buf_len == 0) { return -1; }

    char *p   = buf;
    size_t rem = buf_len;
    int n;

#define _WRITE(...)                          \
    do {                                     \
        n = snprintf(p, rem, __VA_ARGS__);   \
        if (n < 0 || (size_t)n >= rem) {     \
            return -1;                       \
        }                                    \
        p   += n;                            \
        rem -= (size_t)n;                    \
    } while (0)

    _WRITE("{");
    _WRITE("\"timestamp\":\"%s\",", timestamp_str);
    _WRITE("\"t_s\":%.3f,", (double)t_s);
    _WRITE("\"T_amb_raw\":%u,", (unsigned)t_amb_raw);
    _WRITE("\"WS_raw\":%u,", (unsigned)ws_raw);
    _WRITE("\"lat\":%.6f,", (double)lat);
    _WRITE("\"lon\":%.6f,", (double)lon);
    _WRITE("\"fault_flags\":%u,", (unsigned)fault_flags);
    _WRITE("\"edge_version\":\"%s\"", edge_version);
    _WRITE("}");

#undef _WRITE

    return (int)(buf_len - rem - 1);   /* characters written, excl. NUL */
}
