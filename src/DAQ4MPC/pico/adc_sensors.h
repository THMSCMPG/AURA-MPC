/* pico/adc_sensors.h -- Raw ADC sampling for AURA-MFP.
 *
 * ARCHITECTURE (confirmed this session): calibration lives on the
 * WORKSTATION, not the firmware -- the Pico is a translator, piping raw
 * sensor values over serial; the workstation does all scaling/prediction.
 * This used to scale to physical units in firmware; now emits raw
 * oversampled 12-bit ADC counts (0-4095) directly. See adc_sensors.c's
 * header comment for the full rationale.
 *
 * pyranometer (G_poa/ADC0) REMOVED -- irradiance is now a manual
 * calibration-time entry on the workstation side, not sensed. GP26 is
 * free. Two channels remain, sampled on every epoch:
 *
 *   ADC1 (GP27)  T_amb   raw count [0, 4095]
 *   ADC2 (GP28)  WS      raw count [0, 4095]
 *
 * NOTE: both of these are also slated to move off analog ADC eventually
 * (T_amb -> weather:bit's BME280 over I2C, WS -> the SparkFun Weather
 * Meter Kit's digital pulse) -- deliberately NOT redesigned yet, since
 * that depends on wiring decisions explicitly deferred until the
 * weather:bit hardware is in hand. Flagged, not acted on prematurely.
 *
 * Each channel is oversampled OVERSAMPLE_COUNT times (see config.h) for
 * noise reduction -- oversampling is a firmware-side concern independent
 * of calibration, stays here.
 */

#ifndef AURA_ADC_SENSORS_H
#define AURA_ADC_SENSORS_H

#include <stdint.h>

/* Initialise both ADC channels.  Must be called once before
 * adc_sensors_read().                                                       */
void adc_sensors_init(void);

/* Sample both ADC channels with oversampling. Returns RAW counts, no
 * scaling or fault detection -- both now happen workstation-side, where
 * the calibration constants actually live.
 *
 * Parameters
 * ----------
 * t_amb_raw_out : output -- raw oversampled ADC1 count [0, 4095]
 * ws_raw_out    : output -- raw oversampled ADC2 count [0, 4095]
 */
void adc_sensors_read(uint16_t *t_amb_raw_out,
                      uint16_t *ws_raw_out);

#endif /* AURA_ADC_SENSORS_H */
