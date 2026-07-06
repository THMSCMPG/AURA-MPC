/* pico/rtc_ds3231.h – Minimal I²C driver for the DS3231 RTC.
 *
 * Only the date/time read path is implemented – the DS3231 alarm,
 * temperature, and EEPROM features are not used by this firmware.
 *
 * If the DS3231 is absent or the I²C transaction fails, get_datetime()
 * returns false and the caller should fall back to elapsed milliseconds
 * from boot (see main.c) and set FAULT_RTC_LOST in fault_flags.
 */

#ifndef AURA_RTC_DS3231_H
#define AURA_RTC_DS3231_H

#include <stdbool.h>
#include <stdint.h>

/* Date/time as returned by the DS3231 (all values calendar-natural). */
typedef struct {
    uint16_t year;    /* e.g. 2026                */
    uint8_t  month;   /* 1–12                     */
    uint8_t  day;     /* 1–31                     */
    uint8_t  hour;    /* 0–23                     */
    uint8_t  minute;  /* 0–59                     */
    uint8_t  second;  /* 0–59                     */
} rtc_datetime_t;

/* Initialise the I²C bus and probe for the DS3231.
 * Returns true if the device acknowledges.                              */
bool rtc_ds3231_init(void);

/* Read the current date/time from the DS3231.
 * Returns true on success; false if the device did not respond or
 * returned corrupt BCD data.                                            */
bool rtc_ds3231_get_datetime(rtc_datetime_t *dt_out);

#endif /* AURA_RTC_DS3231_H */
