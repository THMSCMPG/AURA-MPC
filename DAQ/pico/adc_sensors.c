/* pico/adc_sensors.c – ADC read + physical-unit scaling for AURA-MFP.
 *
 * ── Calibration constants ────────────────────────────────────────────────
 * Tune the ADCn_V_MIN / ADCn_V_MAX / ADCn_PHYS_MIN / ADCn_PHYS_MAX defines
 * below to match your sensor wiring.  V_MIN/V_MAX are the voltages that
 * appear on the RP2040 ADC pin (after any voltage dividers), while
 * PHYS_MIN/PHYS_MAX are the corresponding physical-unit endpoints.
 *
 * ADC0 – G_poa pyranometer (conditioned to 0–3.3 V ≡ 0–1200 W/m²)
 *   If your sensor outputs 0–5 V use a resistor divider (e.g. 33 kΩ/22 kΩ)
 *   to bring it to 0–3.3 V before connecting to GP26.
 *
 * ADC1 – T_amb thermocouple signal conditioner (0–3.3 V ≡ −20–+60 °C)
 *
 * ADC2 – WS anemometer analogue output (0–3.3 V ≡ 0–20 m/s)
 */

#include "adc_sensors.h"
#include "config.h"
#include "fault_flags.h"

#include "hardware/adc.h"

#include <stdint.h>

/* ── Calibration constants (edit here to retune) ─────────────────────── */

/* ADC0 – G_poa irradiance */
#define ADC0_V_MIN      0.0f       /* V  – voltage at sensor minimum      */
#define ADC0_V_MAX      3.3f       /* V  – voltage at sensor maximum      */
#define ADC0_PHYS_MIN   0.0f       /* W/m² at ADC0_V_MIN                  */
#define ADC0_PHYS_MAX   1200.0f    /* W/m² at ADC0_V_MAX                  */

/* ADC1 – T_amb ambient temperature */
#define ADC1_V_MIN      0.0f       /* V  – voltage at sensor minimum      */
#define ADC1_V_MAX      3.3f       /* V  – voltage at sensor maximum      */
#define ADC1_PHYS_MIN  -20.0f      /* °C at ADC1_V_MIN                    */
#define ADC1_PHYS_MAX   60.0f      /* °C at ADC1_V_MAX                    */

/* ADC2 – WS wind speed */
#define ADC2_V_MIN      0.0f       /* V  – voltage at sensor minimum      */
#define ADC2_V_MAX      3.3f       /* V  – voltage at sensor maximum      */
#define ADC2_PHYS_MIN   0.0f       /* m/s at ADC2_V_MIN                   */
#define ADC2_PHYS_MAX   20.0f      /* m/s at ADC2_V_MAX                   */

/* RP2040 ADC reference voltage and full-scale counts */
#define ADC_VREF        3.3f
#define ADC_FULL_SCALE  4095.0f    /* 12-bit ADC */

/* ── Internal helpers ─────────────────────────────────────────────────── */

/* Convert a 12-bit ADC count to voltage (referenced to ADC_VREF). */
static inline float _counts_to_volts(uint16_t counts)
{
    return (counts / ADC_FULL_SCALE) * ADC_VREF;
}

/* Linear interpolation: map voltage v from [v_min, v_max] to
 * [phys_min, phys_max].  Returns the clamped physical value and sets
 * *oor if the raw voltage was outside the calibrated range.             */
static float _scale(float v, float v_min, float v_max,
                    float phys_min, float phys_max, int *oor)
{
    float t = (v - v_min) / (v_max - v_min);
    if (t < 0.0f) { t = 0.0f; *oor = 1; }
    if (t > 1.0f) { t = 1.0f; *oor = 1; }
    return phys_min + t * (phys_max - phys_min);
}

/* Read one ADC channel OVERSAMPLE_COUNT times and return the average
 * voltage.                                                              */
static float _oversample_channel(uint adc_channel)
{
    uint32_t acc = 0;
    adc_select_input(adc_channel);
    for (int i = 0; i < OVERSAMPLE_COUNT; i++) {
        acc += adc_read();
    }
    uint16_t avg_counts = (uint16_t)(acc / OVERSAMPLE_COUNT);
    return _counts_to_volts(avg_counts);
}

/* ── Public API ──────────────────────────────────────────────────────── */

void adc_sensors_init(void)
{
    adc_init();
    adc_gpio_init(26);   /* GP26 = ADC0 – G_poa  */
    adc_gpio_init(27);   /* GP27 = ADC1 – T_amb  */
    adc_gpio_init(28);   /* GP28 = ADC2 – WS     */
}

void adc_sensors_read(float *g_poa_out,
                      float *t_amb_out,
                      float *ws_out,
                      uint16_t *fault_flags_out)
{
    int oor;

    /* ADC0 – G_poa */
    float v0 = _oversample_channel(0);
    oor = 0;
    *g_poa_out = _scale(v0, ADC0_V_MIN, ADC0_V_MAX,
                        ADC0_PHYS_MIN, ADC0_PHYS_MAX, &oor);
    if (oor) { *fault_flags_out |= FAULT_ADC0_OOR; }

    /* ADC1 – T_amb */
    float v1 = _oversample_channel(1);
    oor = 0;
    *t_amb_out = _scale(v1, ADC1_V_MIN, ADC1_V_MAX,
                        ADC1_PHYS_MIN, ADC1_PHYS_MAX, &oor);
    if (oor) { *fault_flags_out |= FAULT_ADC1_OOR; }

    /* ADC2 – WS */
    float v2 = _oversample_channel(2);
    oor = 0;
    *ws_out = _scale(v2, ADC2_V_MIN, ADC2_V_MAX,
                     ADC2_PHYS_MIN, ADC2_PHYS_MAX, &oor);
    if (oor) { *fault_flags_out |= FAULT_ADC2_OOR; }
}
