/* pico/main.c – AURA-MFP Pico acquisition firmware entry point.
 *
 * Sample loop (every SAMPLE_CADENCE_MS):
 *
 *   1. Read timestamp from DS3231 RTC over I²C.
 *      On failure: use elapsed milliseconds from boot, set FAULT_RTC_LOST.
 *   2. Read all three ADC channels (with 16× oversampling each).
 *      Out-of-range readings: clamp + set FAULT_ADCn_OOR bit.
 *   3. Build a PINN_SENSOR_PACKET_SCHEMA JSON string.
 *   4. Emit the JSON line over UART0 (GP0/GP1) and USB CDC.
 *   5. Blink the on-board LED (GP25) once to confirm successful emit.
 *   6. Sleep until the next epoch.
 *
 * All compile-time knobs live in config.h.
 * All fault-flag bit definitions live in fault_flags.h.
 */

#include "config.h"
#include "fault_flags.h"
#include "adc_sensors.h"
#include "rtc_ds3231.h"
#include "json_builder.h"
#include "uart_output.h"

#include "pico/stdlib.h"
#include "hardware/gpio.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ── Internal helpers ─────────────────────────────────────────────────── */

/* Build the ISO-8601 timestamp string and compute t_s (seconds-of-day).
 *
 * If the RTC is available, uses its date/time.  Otherwise falls back to
 * elapsed milliseconds from boot, emitting a synthetic timestamp anchored
 * to 1970-01-01T00:00:00Z (UNIX epoch) plus the boot offset.
 *
 * timestamp_out must point to a buffer of at least 28 bytes.
 */
static float _get_timestamp(bool rtc_ok,
                             const rtc_datetime_t *dt,
                             uint32_t boot_ms,
                             char *timestamp_out)
{
    float t_s;

    if (rtc_ok) {
        /* Format: YYYY-MM-DDTHH:MM:SS.000Z */
        snprintf(timestamp_out, 28,
                 "%04u-%02u-%02uT%02u:%02u:%02u.000Z",
                 (unsigned)dt->year,
                 (unsigned)dt->month,
                 (unsigned)dt->day,
                 (unsigned)dt->hour,
                 (unsigned)dt->minute,
                 (unsigned)dt->second);
        /* t_s = seconds-of-day */
        t_s = (float)(dt->hour * 3600u + dt->minute * 60u + dt->second);
    } else {
        /* Fallback: elapsed milliseconds from boot, shown as time-of-day.
         * Both the timestamp and t_s use the same modulo-86400 value so
         * they stay consistent even after the device has been running for
         * more than 24 hours.                                             */
        uint32_t secs     = boot_ms / 1000u;
        uint32_t ms       = boot_ms % 1000u;
        uint32_t secs_day = secs % 86400u;
        snprintf(timestamp_out, 28,
                 "1970-01-01T%02u:%02u:%02u.%03uZ",
                 (unsigned)(secs_day / 3600u),
                 (unsigned)((secs_day / 60u) % 60u),
                 (unsigned)(secs_day % 60u),
                 (unsigned)ms);
        t_s = (float)secs_day;
    }

    return t_s;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    /* Initialise Pico SDK stdio (USB CDC + any other configured sink). */
    stdio_init_all();

    /* On-board LED. */
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);

    /* Peripherals. */
    uart_output_init();
    adc_sensors_init();

    bool rtc_ok = rtc_ds3231_init();

    char          timestamp_str[28];
    char          json_buf[JSON_BUF_SIZE];
    rtc_datetime_t dt;

    while (1) {
        uint32_t t0_ms   = to_ms_since_boot(get_absolute_time());
        uint16_t fault_flags = 0;

        /* ── RTC timestamp ──────────────────────────────────────────── */
        if (rtc_ok) {
            rtc_ok = rtc_ds3231_get_datetime(&dt);
        }
        if (!rtc_ok) {
            fault_flags |= FAULT_RTC_LOST;
        }
        float t_s = _get_timestamp(rtc_ok, rtc_ok ? &dt : NULL,
                                   t0_ms, timestamp_str);

        /* ── ADC sensors (oversampled + scaled) ─────────────────────── */
        float g_poa, t_amb, ws;
        adc_sensors_read(&g_poa, &t_amb, &ws, &fault_flags);

        /* ── Build JSON packet ──────────────────────────────────────── */
        int json_len = json_build_packet(
            json_buf, sizeof(json_buf),
            timestamp_str,
            t_s, g_poa, t_amb, ws,
            DEPLOY_LAT, DEPLOY_LON,
            fault_flags,
            EDGE_VERSION);

        if (json_len > 0) {
            /* ── Emit ───────────────────────────────────────────────── */
            uart_output_emit(json_buf);

            /* Blink LED to indicate successful packet emit. */
            gpio_put(LED_PIN, 1);
            sleep_ms(50);
            gpio_put(LED_PIN, 0);
        }

        /* ── Sleep until next epoch ─────────────────────────────────── */
        uint32_t elapsed_ms = to_ms_since_boot(get_absolute_time()) - t0_ms;
        if (elapsed_ms < SAMPLE_CADENCE_MS) {
            sleep_ms(SAMPLE_CADENCE_MS - elapsed_ms);
        }
    }

    return 0;   /* unreachable */
}
