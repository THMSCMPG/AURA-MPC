/* pico/json_builder.h -- Build the Pico's RAW packet JSON strings.
 *
 * ARCHITECTURE (confirmed this session): the Pico is a translator, not a
 * processor -- it emits RAW sensor readings, no calibration or unit
 * conversion. The workstation applies calibration (see
 * DAQ4MPC/workstation/calibration.py's CalibrationLoader) and translates this raw
 * packet into the existing PINN_SENSOR_PACKET_SCHEMA (physical units)
 * BEFORE anything reaches decision_server.py -- so decision_server.py,
 * edge_adapter.py, and everything downstream of the translation step
 * needed ZERO changes for this. Only the wire format and the new
 * workstation-side translation step are new.
 *
 * PICO_RAW_PACKET_SCHEMA (this is what actually goes over the wire now):
 *
 *   { "timestamp": "<ISO-8601>", "t_s": <float>,
 *     "T_amb_raw": <int, 0-4095>, "WS_raw": <int, 0-4095>,
 *     "lat": <float>, "lon": <float>,
 *     "fault_flags": <uint>, "edge_version": "<str>" }
 *
 * G_poa, CC, sky_image_path, and pose are deliberately NOT part of this
 * raw schema at all (they were always null/workstation-supplied even in
 * the old PINN_SENSOR_PACKET_SCHEMA -- no reason for the Pico to round-trip
 * placeholders for fields it never had data for).
 *
 * The timestamp string uses the format produced by build_sensor_packet():
 *   YYYY-MM-DDTHH:MM:SS.mmmZ   (millisecond precision, trailing Z)
 *
 * t_s is seconds-of-day derived from the RTC (or elapsed seconds from
 * boot when the RTC is unavailable).
 */

#ifndef AURA_JSON_BUILDER_H
#define AURA_JSON_BUILDER_H

#include <stdint.h>
#include <stddef.h>

/* Fill *buf (capacity buf_len) with a NUL-terminated JSON raw-packet string.
 *
 * Parameters
 * ----------
 * buf          : output character buffer
 * buf_len      : size of buf in bytes (must be >= JSON_BUF_SIZE)
 * timestamp_str: ISO-8601 timestamp "YYYY-MM-DDTHH:MM:SS.mmmZ"
 * t_s          : seconds-of-day (or seconds from boot on RTC fault)
 * t_amb_raw    : raw ADC1 count [0, 4095] -- no scaling applied
 * ws_raw       : raw ADC2 count [0, 4095] -- no scaling applied
 * lat          : deployment latitude  [deg]
 * lon          : deployment longitude [deg]
 * fault_flags  : uint16 bitmask
 * edge_version : firmware version string
 *
 * Returns the number of characters written (excluding NUL), or a
 * negative value if buf_len was too small.
 */
int json_build_packet(char *buf, size_t buf_len,
                      const char *timestamp_str,
                      float t_s,
                      uint16_t t_amb_raw, uint16_t ws_raw,
                      float lat,   float lon,
                      uint16_t fault_flags,
                      const char *edge_version);

#endif /* AURA_JSON_BUILDER_H */
