/* pico/fault_flags.h – Bitmask constants for the AURA-MFP Pico firmware.
 *
 * ARCHITECTURE NOTE (this session): calibration -- and therefore
 * physical-range fault detection -- moved to the workstation (see
 * adc_sensors.c). The firmware no longer sets FAULT_ADC1_OOR/
 * FAULT_ADC2_OOR itself (it emits raw counts, with no calibration
 * constants to check a physical range against). These bit definitions
 * stay here in case the workstation wants to reuse the same convention
 * in its own local fault tracking after applying calibration -- just
 * not set by the firmware on the wire packet anymore.
 *
 * These bit positions are used in the ``fault_flags`` field of every emitted
 * packet and must stay stable across firmware versions so downstream
 * consumers can interpret them without recompiling.
 *
 * Bit layout
 * ----------
 *   bit 0  (0x0001)  RESERVED — was ADC0 (G_poa/pyranometer) out-of-range;
 *                    pyranometer removed (irradiance is now manual-entry
 *                    on the workstation, not sensed). Left unused rather
 *                    than reassigned, so any old logged data with this bit
 *                    set is still unambiguous.
 *   bit 1  (0x0002)  ADC1 (T_amb)  -- NOT set by firmware anymore, see above
 *   bit 2  (0x0004)  ADC2 (WS)     -- NOT set by firmware anymore, see above
 *   bit 5  (0x0020)  RTC sync lost (DS3231 absent or I²C error; timestamp
 *                    derived from elapsed milliseconds since boot instead)
 *
 * Bits 3, 4, 6–15 are reserved for future use and shall be zero.
 */

#ifndef AURA_FAULT_FLAGS_H
#define AURA_FAULT_FLAGS_H

#include <stdint.h>

#define FAULT_ADC1_OOR   ((uint16_t)0x0002)  /* reserved for workstation-side use */
#define FAULT_ADC2_OOR   ((uint16_t)0x0004)  /* reserved for workstation-side use */
#define FAULT_RTC_LOST   ((uint16_t)0x0020)  /* RTC sync lost       */

#endif /* AURA_FAULT_FLAGS_H */
