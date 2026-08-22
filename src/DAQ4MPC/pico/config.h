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
 * workstation/packet_builder.py.                                                     */
#define EDGE_VERSION  "v0.1.0"

/* ── UART0 (JSON stream output -> SparkFun OpenLog, local black-box log) ── */
/* USB CDC is now the PRIMARY link to the workstation (Pico-only design,
 * confirmed this session -- USB serial instead of a Pi/Pico W network
 * link). UART0 here is the SECONDARY sink, wired to OpenLog's RXI pin for
 * local logging independent of the workstation connection. See
 * uart_output.h for the emit-to-both-sinks logic. */
#define JSON_UART_ID   uart0
#define JSON_TX_PIN    0    /* GP0 -> OpenLog RXI */
#define JSON_RX_PIN    1    /* GP1 (unused -- OpenLog doesn't talk back) */
#define JSON_BAUD      115200

/* ── ADC (GP26 free -- pyranometer/G_poa removed, irradiance is now
 *      manual-entry on the workstation) ──────────────────────────────── */
/* T_amb (GP27, ADC1) and WS (GP28, ADC2) remain analog for now -- both
 * are slated to move off ADC eventually (T_amb -> weather:bit's BME280
 * over I2C, WS -> the SparkFun Weather Meter Kit's digital pulse), not
 * yet redesigned pending wiring decisions once that hardware is in hand.
 * See adc_sensors.h. */

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
