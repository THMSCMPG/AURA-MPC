/* pico/json_builder.c – Build PINN_SENSOR_PACKET_SCHEMA JSON strings.
 *
 * Produces the exact same field order and encoding as the Python
 * build_sensor_packet() function in pi/packet_builder.py.
 */

#include "json_builder.h"

#include <math.h>
#include <stdio.h>

/* Helper: append either a JSON float with one decimal place, or "null"
 * when the value is NaN.                                               */
static int _append_float_or_null(char *buf, size_t remaining,
                                 float value, int *wrote)
{
    int n;
    if (isnan(value)) {
        n = snprintf(buf, remaining, "null");
    } else {
        /* One decimal place matches the Python repr for sensor data. */
        n = snprintf(buf, remaining, "%.1f", (double)value);
    }
    *wrote = n;
    return n;
}

int json_build_packet(char *buf, size_t buf_len,
                      const char *timestamp_str,
                      float t_s,
                      float g_poa, float t_amb, float ws,
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

    /* G_poa */
    _WRITE("\"G_poa\":");
    int fw;
    _append_float_or_null(p, rem, g_poa, &fw);
    if (fw < 0 || (size_t)fw >= rem) { return -1; }
    p += fw; rem -= (size_t)fw;
    _WRITE(",");

    /* T_amb */
    _WRITE("\"T_amb\":");
    _append_float_or_null(p, rem, t_amb, &fw);
    if (fw < 0 || (size_t)fw >= rem) { return -1; }
    p += fw; rem -= (size_t)fw;
    _WRITE(",");

    /* WS */
    _WRITE("\"WS\":");
    _append_float_or_null(p, rem, ws, &fw);
    if (fw < 0 || (size_t)fw >= rem) { return -1; }
    p += fw; rem -= (size_t)fw;
    _WRITE(",");

    _WRITE("\"CC\":null,");
    _WRITE("\"lat\":%.6f,", (double)lat);
    _WRITE("\"lon\":%.6f,", (double)lon);
    _WRITE("\"sky_image_path\":null,");
    _WRITE("\"pose\":null,");
    _WRITE("\"fault_flags\":%u,", (unsigned)fault_flags);
    _WRITE("\"edge_version\":\"%s\"", edge_version);
    _WRITE("}");

#undef _WRITE

    return (int)(buf_len - rem - 1);   /* characters written, excl. NUL */
}
