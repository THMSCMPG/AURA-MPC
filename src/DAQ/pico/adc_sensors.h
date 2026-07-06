/* pico/adc_sensors.h – ADC read + physical-unit scaling for AURA-MFP.
 *
 * Three channels are sampled on every epoch:
 *
 *   ADC0 (GP26)  G_poa   plane-of-array irradiance   [W/m²]
 *   ADC1 (GP27)  T_amb   ambient air temperature      [°C]
 *   ADC2 (GP28)  WS      local wind speed             [m/s]
 *
 * Each channel is oversampled OVERSAMPLE_COUNT times (see config.h) and
 * the average is converted to a physical value via a linear mapping:
 *
 *   phys = PHYS_MIN + (v_avg - V_MIN) / (V_MAX - V_MIN) * (PHYS_MAX - PHYS_MIN)
 *
 * Calibration constants are defined in adc_sensors.c and can be tuned
 * without recompiling the rest of the firmware.
 *
 * Out-of-range detection
 * ----------------------
 * After scaling, each channel's physical value is checked against
 * [PHYS_MIN, PHYS_MAX].  Readings outside this range are clamped and the
 * corresponding FAULT_ADCn_OOR bit is set in *fault_flags_out.
 */

#ifndef AURA_ADC_SENSORS_H
#define AURA_ADC_SENSORS_H

#include <stdint.h>

/* Initialise all three ADC channels.  Must be called once before
 * adc_sensors_read().                                                       */
void adc_sensors_init(void);

/* Sample all three ADC channels with oversampling and scale to physical
 * units.
 *
 * Parameters
 * ----------
 * g_poa_out      : output – irradiance reading [W/m²]
 * t_amb_out      : output – temperature reading [°C]
 * ws_out         : output – wind speed reading  [m/s]
 * fault_flags_out: input/output – FAULT_ADCn_OOR bits ORed in on fault
 */
void adc_sensors_read(float *g_poa_out,
                      float *t_amb_out,
                      float *ws_out,
                      uint16_t *fault_flags_out);

#endif /* AURA_ADC_SENSORS_H */
