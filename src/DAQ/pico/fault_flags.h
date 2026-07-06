/* pico/fault_flags.h – Bitmask constants for the AURA-MFP Pico firmware.
 *
 * These bit positions are used in the ``fault_flags`` field of every emitted
 * PINN_SENSOR_PACKET_SCHEMA JSON packet and must stay stable across firmware
 * versions so that the Pi-side daemon can interpret them without recompiling.
 *
 * Bit layout
 * ----------
 *   bit 0  (0x0001)  ADC0 (G_poa)  reading out of calibrated range
 *   bit 1  (0x0002)  ADC1 (T_amb)  reading out of calibrated range
 *   bit 2  (0x0004)  ADC2 (WS)     reading out of calibrated range
 *   bit 5  (0x0020)  RTC sync lost (DS3231 absent or I²C error; timestamp
 *                    derived from elapsed milliseconds since boot instead)
 *
 * Bits 3, 4, 6–15 are reserved for future use and shall be zero.
 */

#ifndef AURA_FAULT_FLAGS_H
#define AURA_FAULT_FLAGS_H

#include <stdint.h>

#define FAULT_ADC0_OOR   ((uint16_t)0x0001)  /* G_poa out-of-range  */
#define FAULT_ADC1_OOR   ((uint16_t)0x0002)  /* T_amb out-of-range  */
#define FAULT_ADC2_OOR   ((uint16_t)0x0004)  /* WS    out-of-range  */
#define FAULT_RTC_LOST   ((uint16_t)0x0020)  /* RTC sync lost       */

#endif /* AURA_FAULT_FLAGS_H */
