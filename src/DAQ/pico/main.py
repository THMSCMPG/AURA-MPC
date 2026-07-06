"""pico/main.py – Entry point for the EDGE-AURA-MFP Pico firmware.

Sample loop (every ``SAMPLE_INTERVAL_MS``):

    1. Pet the watchdog.
    2. Read each sensor via its driver's ``read()``.
       On exception → set fault flag for that sensor, substitute the
       protocol sentinel value into the payload, bump the consecutive-
       error counter.  If any sensor exceeds
       ``FAULT_PERSIST_THRESHOLD`` consecutive errors, latch a
       persistent fault (bit 0x8000 in ``fault_flags``) and append a
       one-line record to flash – once per sensor per boot to avoid
       wearing out the flash.
    3. Build one aggregated payload, pack into a v1 frame (40 B),
       COBS-wrap, write over UART0 to the Pi.
    4. Sleep until next interval.

Hardware setup is centralised in ``pico/config.py``; swapping a sensor
means swapping one driver file – no changes here.
"""

import time

try:  # pragma: no cover – only present on MicroPython
    from machine import I2C, SPI, UART, Pin  # type: ignore
    _MICROPYTHON = True
except ImportError:
    _MICROPYTHON = False

import config                                          # noqa: E402
from watchdog import Watchdog                          # noqa: E402
from drivers.pyranometer_adc       import PyranometerADC        # noqa: E402
from drivers.thermocouple_max31856 import ThermocoupleMAX31856  # noqa: E402
from drivers.anemometer_rs485      import AnemometerRS485       # noqa: E402
from drivers.rtc_ds3231            import RtcDS3231              # noqa: E402
from drivers.sensor_base           import SensorError            # noqa: E402
from protocol import framing, packet                             # noqa: E402


# ── Fault-flag bit positions (mirrors packet.FAULT_*) ─────────────────────
_FAULT_PYR = packet.FAULT_PYRANOMETER
_FAULT_TC  = packet.FAULT_TC_BITS
_FAULT_ANE = packet.FAULT_ANEMOMETER
_FAULT_RTC = packet.FAULT_RTC
_FAULT_PER = packet.FAULT_PERSISTENT


def _init_hardware():
    """Create and return all driver instances + host UART. Dev-mode safe."""
    if not _MICROPYTHON:
        return None, PyranometerADC(), ThermocoupleMAX31856(), AnemometerRS485(), RtcDS3231()

    i2c = I2C(config.I2C_ID,
              sda=Pin(config.I2C_SDA_PIN), scl=Pin(config.I2C_SCL_PIN),
              freq=config.I2C_FREQ)
    spi = SPI(config.SPI_ID, baudrate=config.SPI_BAUD,
              sck=Pin(config.SPI_SCK_PIN),
              mosi=Pin(config.SPI_MOSI_PIN),
              miso=Pin(config.SPI_MISO_PIN))
    rs485_uart = UART(config.RS485_UART_ID, baudrate=config.RS485_BAUD,
                      tx=Pin(config.RS485_TX_PIN), rx=Pin(config.RS485_RX_PIN))
    host_uart = UART(config.HOST_UART_ID, baudrate=config.HOST_BAUD,
                     tx=Pin(config.HOST_TX_PIN), rx=Pin(config.HOST_RX_PIN))

    pyr = PyranometerADC(adc_pin=config.ADC_PIN)
    tc  = ThermocoupleMAX31856(spi=spi, cs_pins=config.SPI_CS_PINS)
    ane = AnemometerRS485(uart=rs485_uart, de_pin=config.RS485_DE_PIN,
                          slave_addr=config.RS485_MODBUS_ADDR)
    rtc = RtcDS3231(i2c=i2c)
    return host_uart, pyr, tc, ane, rtc


def _log_persistent_fault(tag: str, logged: dict) -> None:
    """Append one line to flash the first time a sensor latches persistent.

    ``logged`` is a boot-session dict so we never log the same sensor twice.
    """
    if logged.get(tag):
        return
    logged[tag] = True
    try:
        with open(config.FAULT_LOG_PATH, "a") as fh:
            fh.write("%d %s persistent_fault\n" % (int(time.time()), tag))
    except OSError:
        pass  # flash full / read-only FS – nothing we can do


def main():
    host_uart, pyr, tc, ane, rtc = _init_hardware()

    wdt = Watchdog(timeout_ms=config.WDT_TIMEOUT_MS)
    wdt.start()

    logged_persistent: dict = {}
    threshold = config.FAULT_PERSIST_THRESHOLD

    while True:
        wdt.feed()
        fault_flags = 0

        # ── Timestamp (RTC) ────────────────────────────────────────────
        try:
            ts_ms = rtc.read()
        except SensorError:
            fault_flags |= _FAULT_RTC
            ts_ms = int(time.time() * 1000) if not _MICROPYTHON else 0
            if rtc.consecutive_errors >= threshold:
                fault_flags |= _FAULT_PER
                _log_persistent_fault("rtc", logged_persistent)

        # ── Pyranometer ────────────────────────────────────────────────
        try:
            pyr_raw = pyr.read()
        except SensorError:
            fault_flags |= _FAULT_PYR
            pyr_raw = packet.PYRANOMETER_FAULT_SENTINEL
            if pyr.consecutive_errors >= threshold:
                fault_flags |= _FAULT_PER
                _log_persistent_fault("pyranometer", logged_persistent)

        # ── Thermocouple (×4 channels) ─────────────────────────────────
        try:
            tc_raw = tc.read()
        except SensorError:
            tc_raw = [packet.THERMOCOUPLE_FAULT_SENTINEL] * 4
            fault_flags |= (_FAULT_TC[0] | _FAULT_TC[1] | _FAULT_TC[2] | _FAULT_TC[3])
            if tc.consecutive_errors >= threshold:
                fault_flags |= _FAULT_PER
                _log_persistent_fault("thermocouple", logged_persistent)
        else:
            for i, raw in enumerate(tc_raw):
                if raw == packet.THERMOCOUPLE_FAULT_SENTINEL:
                    fault_flags |= _FAULT_TC[i]

        # ── Anemometer ─────────────────────────────────────────────────
        try:
            speed_x100, direction = ane.read()
        except SensorError:
            fault_flags |= _FAULT_ANE
            speed_x100 = packet.ANEMOMETER_SPEED_FAULT_SENTINEL
            direction  = packet.ANEMOMETER_DIR_FAULT_SENTINEL
            if ane.consecutive_errors >= threshold:
                fault_flags |= _FAULT_PER
                _log_persistent_fault("anemometer", logged_persistent)

        # ── Build + emit frame ─────────────────────────────────────────
        frame = packet.pack(
            timestamp_ms=ts_ms,
            pyranometer_raw=pyr_raw,
            thermocouple_raw=tc_raw,
            anemometer_speed_x100=speed_x100,
            anemometer_dir_deg=direction,
            fault_flags=fault_flags,
        )
        wire = framing.wrap(frame)
        if host_uart is not None:
            host_uart.write(wire)

        if _MICROPYTHON:
            time.sleep_ms(config.SAMPLE_INTERVAL_MS)  # type: ignore[attr-defined]
        else:                                         # pragma: no cover
            time.sleep(config.SAMPLE_INTERVAL_MS / 1000.0)


if __name__ == "__main__":
    main()
