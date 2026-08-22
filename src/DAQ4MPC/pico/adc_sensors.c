/* pico/adc_sensors.c -- Raw ADC sampling for AURA-MFP.
 *
 * ARCHITECTURE (confirmed this session): calibration lives on the
 * WORKSTATION, not the firmware. The Pico is a translator -- it reads
 * sensors and pipes raw values over serial; the workstation does all
 * scaling, calibration, and prediction. This file used to do raw-count
 * -> physical-unit scaling in firmware (fixed PHYS_MIN/PHYS_MAX constants
 * baked in at compile time, meaning recalibrating required a reflash).
 * That's been removed -- this now emits raw oversampled 12-bit ADC
 * counts (0-4095) directly. Workstation-side calibration.py's
 * CalibrationLoader applies slope/intercept from calibration/*.json,
 * fitted by the calibrate.py wizard -- no reflash needed to recalibrate.
 *
 * pyranometer (G_poa) REMOVED separately -- irradiance is now a manual
 * calibration-time entry on the workstation side, not sensed at all.
 * ADC0/GP26 is freed up.
 *
 * ADC1 (GP27) -- T_amb thermocouple signal conditioner, raw count
 * ADC2 (GP28) -- WS anemometer analogue output, raw count
 *
 * NOTE: both of these are also slated to move off analog ADC eventually
 * (T_amb -> weather:bit's BME280 over I2C, WS -> the SparkFun Weather
 * Meter Kit's digital pulse) -- deliberately NOT redesigned yet, since
 * that depends on wiring decisions explicitly deferred until the
 * weather:bit hardware is in hand. Flagging so it isn't forgotten, not
 * acting on it prematurely.
 *
 * Out-of-range/fault detection also moves to the workstation (it needs
 * calibration to know what "out of range" even means in physical units)
 * -- firmware no longer sets FAULT_ADC1_OOR/FAULT_ADC2_OOR. Those bit
 * definitions stay in fault_flags.h in case the workstation wants to use
 * the same convention in its own local fault tracking, just not on the
 * wire packet the firmware builds.
 */

#include "adc_sensors.h"
#include "config.h"

#include "hardware/adc.h"

#include <stdint.h>

/* RP2040 ADC full-scale count (12-bit) -- NOT a calibration constant,
 * just the hardware's own native resolution. Emitted raw counts are in
 * [0, ADC_FULL_SCALE_COUNTS]. */
#define ADC_FULL_SCALE_COUNTS  4095u

/* ── Internal helpers ─────────────────────────────────────────────────── */

/* Read one ADC channel OVERSAMPLE_COUNT times and return the averaged
 * raw count. Oversampling is a firmware-side noise-reduction concern,
 * independent of calibration -- stays here. */
static uint16_t _oversample_channel(uint adc_channel)
{
    uint32_t acc = 0;
    adc_select_input(adc_channel);
    for (int i = 0; i < OVERSAMPLE_COUNT; i++) {
        acc += adc_read();
    }
    return (uint16_t)(acc / OVERSAMPLE_COUNT);
}

/* ── Public API ──────────────────────────────────────────────────────── */

void adc_sensors_init(void)
{
    adc_init();
    adc_gpio_init(27);   /* GP27 = ADC1 -- T_amb  */
    adc_gpio_init(28);   /* GP28 = ADC2 -- WS     */
}

void adc_sensors_read(uint16_t *t_amb_raw_out,
                      uint16_t *ws_raw_out)
{
    *t_amb_raw_out = _oversample_channel(1);
    *ws_raw_out    = _oversample_channel(2);
}
