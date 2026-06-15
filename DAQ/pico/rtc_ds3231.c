/* pico/rtc_ds3231.c – Minimal I²C driver for the DS3231 RTC.
 *
 * The DS3231 stores time in BCD registers at I²C address 0x68.
 * Register map (read 7 bytes starting at 0x00):
 *
 *   0x00  seconds   (BCD, 0–59)
 *   0x01  minutes   (BCD, 0–59)
 *   0x02  hours     (BCD, 0–23, 24-hour mode assumed)
 *   0x03  day-of-week (1–7, not used)
 *   0x04  date      (BCD, 1–31)
 *   0x05  month     (BCD, 1–12, century bit in bit 7 ignored)
 *   0x06  year      (BCD, 0–99, offset from 2000)
 */

#include "rtc_ds3231.h"
#include "config.h"

#include "hardware/i2c.h"
#include "pico/stdlib.h"

#include <string.h>

#define DS3231_ADDR   0x68
#define DS3231_REG0   0x00   /* first time/date register */
#define DS3231_NBYTES 7      /* registers 0x00..0x06     */

/* BCD to binary conversion. */
static inline uint8_t _bcd2bin(uint8_t bcd)
{
    return (uint8_t)(((bcd >> 4) & 0x0F) * 10 + (bcd & 0x0F));
}

bool rtc_ds3231_init(void)
{
    i2c_init(RTC_I2C_PORT, RTC_I2C_FREQ);
    gpio_set_function(RTC_SDA_PIN, GPIO_FUNC_I2C);
    gpio_set_function(RTC_SCL_PIN, GPIO_FUNC_I2C);
    gpio_pull_up(RTC_SDA_PIN);
    gpio_pull_up(RTC_SCL_PIN);

    /* Probe: write the register pointer only; no data. */
    uint8_t reg = DS3231_REG0;
    int ret = i2c_write_blocking(RTC_I2C_PORT, DS3231_ADDR, &reg, 1, false);
    return (ret == 1);
}

bool rtc_ds3231_get_datetime(rtc_datetime_t *dt_out)
{
    if (!dt_out) { return false; }

    /* Point to register 0x00. */
    uint8_t reg = DS3231_REG0;
    if (i2c_write_blocking(RTC_I2C_PORT, DS3231_ADDR, &reg, 1, true) != 1) {
        return false;
    }

    uint8_t buf[DS3231_NBYTES];
    if (i2c_read_blocking(RTC_I2C_PORT, DS3231_ADDR,
                          buf, DS3231_NBYTES, false) != DS3231_NBYTES) {
        return false;
    }

    dt_out->second = _bcd2bin(buf[0] & 0x7F);
    dt_out->minute = _bcd2bin(buf[1] & 0x7F);
    dt_out->hour   = _bcd2bin(buf[2] & 0x3F);   /* mask 12/24-hr bit */
    /* buf[3] = day-of-week, skip */
    dt_out->day    = _bcd2bin(buf[4] & 0x3F);
    dt_out->month  = _bcd2bin(buf[5] & 0x1F);   /* mask century bit  */
    dt_out->year   = (uint16_t)(2000 + _bcd2bin(buf[6]));

    /* Basic sanity check. */
    if (dt_out->second > 59 || dt_out->minute > 59 || dt_out->hour > 23 ||
        dt_out->day   < 1  || dt_out->day   > 31  ||
        dt_out->month < 1  || dt_out->month > 12) {
        return false;
    }

    return true;
}
