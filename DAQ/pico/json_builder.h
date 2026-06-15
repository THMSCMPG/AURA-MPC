/* pico/json_builder.h – Build PINN_SENSOR_PACKET_SCHEMA JSON strings.
 *
 * Produces a single-line JSON object whose field order and types exactly
 * match the Python ``PINN_SENSOR_PACKET_SCHEMA`` defined in
 * ``pi/packet_builder.py``:
 *
 *   { "timestamp": "<ISO-8601>", "t_s": <float>,
 *     "G_poa": <float|null>, "T_amb": <float|null>, "WS": <float|null>,
 *     "CC": null, "lat": <float>, "lon": <float>,
 *     "sky_image_path": null, "pose": null,
 *     "fault_flags": <uint>, "edge_version": "<str>" }
 *
 * The timestamp string uses the format produced by build_sensor_packet():
 *   YYYY-MM-DDTHH:MM:SS.mmmZ   (millisecond precision, trailing Z)
 *
 * t_s is seconds-of-day derived from the RTC (or elapsed seconds from
 * boot when the RTC is unavailable).
 *
 * Null handling
 * -------------
 * If G_poa, T_amb, or WS are NaN (tested via isnan()), they are emitted
 * as JSON null.  In normal operation the corresponding FAULT_ADCn_OOR
 * flag is set instead and the clamped physical value is emitted.
 */

#ifndef AURA_JSON_BUILDER_H
#define AURA_JSON_BUILDER_H

#include <stdint.h>
#include <stddef.h>

/* Fill *buf (capacity buf_len) with a NUL-terminated JSON packet string.
 *
 * Parameters
 * ----------
 * buf          : output character buffer
 * buf_len      : size of buf in bytes (must be ≥ JSON_BUF_SIZE)
 * timestamp_str: ISO-8601 timestamp "YYYY-MM-DDTHH:MM:SS.mmmZ"
 * t_s          : seconds-of-day (or seconds from boot on RTC fault)
 * g_poa        : irradiance  [W/m²] — NaN → null
 * t_amb        : temperature [°C]   — NaN → null
 * ws           : wind speed  [m/s]  — NaN → null
 * lat          : deployment latitude  [°]
 * lon          : deployment longitude [°]
 * fault_flags  : uint16 bitmask
 * edge_version : firmware version string
 *
 * Returns the number of characters written (excluding NUL), or a
 * negative value if buf_len was too small.
 */
int json_build_packet(char *buf, size_t buf_len,
                      const char *timestamp_str,
                      float t_s,
                      float g_poa, float t_amb, float ws,
                      float lat,   float lon,
                      uint16_t fault_flags,
                      const char *edge_version);

#endif /* AURA_JSON_BUILDER_H */
