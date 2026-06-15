/* pico/config.h – Compile-time constants for the AURA-MFP Pico firmware.
 *
 * Edit this file to configure deployment coordinates, sample cadence,
 * oversampling depth, and I²C pin assignments.  All hardware-specific
 * tuning lives here; no changes are needed in main.c or the drivers.
 */

#ifndef AURA_PICO_CONFIG_H
#define AURA_PICO_CONFIG_H

/* ── Deployment coordinates (decimal degrees) ──────────────────────────── */
#define DEPLOY_LAT    36.5f
#define DEPLOY_LON   -87.3f

/* ── Sample cadence ─────────────────────────────────────────────────────── */
/* Time between emitted JSON packets, in milliseconds. */
#define SAMPLE_CADENCE_MS  1000

/* Number of raw ADC reads averaged per channel per epoch (noise reduction). */
#define OVERSAMPLE_COUNT   16

/* ── Firmware identity ──────────────────────────────────────────────────── */
/* Must match the EDGE-AURA-MFP release tag and EDGE_VERSION in
 * pi/packet_builder.py.                                                     */
#define EDGE_VERSION  "v0.1.0"

/* ── UART0 (JSON stream output → Raspberry Pi) ──────────────────────────── */
#define JSON_UART_ID   uart0
#define JSON_TX_PIN    0    /* GP0 */
#define JSON_RX_PIN    1    /* GP1 */
#define JSON_BAUD      115200

/* ── I²C0 (DS3231 RTC) ──────────────────────────────────────────────────── */
#define RTC_I2C_PORT   i2c0
#define RTC_SDA_PIN    4    /* GP4 */
#define RTC_SCL_PIN    5    /* GP5 */
#define RTC_I2C_FREQ   400000   /* 400 kHz fast-mode */

/* ── LED (status blink) ─────────────────────────────────────────────────── */
#define LED_PIN        25   /* GP25 = on-board LED */

/* ── JSON output buffer size ────────────────────────────────────────────── */
/* Large enough for the longest possible PINN_SENSOR_PACKET_SCHEMA string. */
#define JSON_BUF_SIZE  512

#endif /* AURA_PICO_CONFIG_H */
