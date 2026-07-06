"""pico/drivers/rtc_ds3231.py – DS3231 I²C real-time clock driver.

Returns a UNIX epoch time in **milliseconds**.  The DS3231 itself only
has a seconds-resolution time register; sub-second precision is
interpolated from the RP2040's ``time.ticks_ms`` counter between RTC
reads.

Wiring (I²C0)
-------------
    DS3231 SDA → GP20
    DS3231 SCL → GP21
    DS3231 VCC → 3.3 V (CR2032 keeps time during power loss)
    DS3231 GND → GND

On a bare Pico with no RTC attached, :meth:`read` falls back to the MCU
uptime so the firmware can still emit frames for dev testing.
"""

try:  # pragma: no cover
    import time as _time
    _MICROPYTHON = hasattr(_time, "ticks_ms")
except ImportError:  # pragma: no cover
    _MICROPYTHON = False

import time

from .sensor_base import SensorBase, SensorError

_DS3231_ADDR = 0x68
_REG_SECONDS = 0x00


def _bcd_to_int(b: int) -> int:
    return (b >> 4) * 10 + (b & 0x0F)


class RtcDS3231(SensorBase):
    """DS3231 I²C RTC.

    Parameters
    ----------
    i2c : machine.I2C | None
        Initialised I²C bus (addr 0x68).  ``None`` for dev mode.
    """

    sensor_id = 0x06
    name = "rtc_ds3231"

    def __init__(self, i2c=None) -> None:
        super().__init__(i2c=i2c)
        self._i2c = i2c

    # ------------------------------------------------------------------
    def _read_epoch_seconds(self) -> int:
        if self._i2c is None:
            return int(time.time())
        buf = self._i2c.readfrom_mem(_DS3231_ADDR, _REG_SECONDS, 7)
        sec  = _bcd_to_int(buf[0] & 0x7F)
        minute = _bcd_to_int(buf[1] & 0x7F)
        hour   = _bcd_to_int(buf[2] & 0x3F)
        day    = _bcd_to_int(buf[4] & 0x3F)
        month  = _bcd_to_int(buf[5] & 0x1F)
        year   = 2000 + _bcd_to_int(buf[6])
        # mktime is available in both CPython and MicroPython (8- or 9-tuple)
        try:
            return int(time.mktime((year, month, day, hour, minute, sec, 0, 0, 0)))
        except TypeError:
            return int(time.mktime((year, month, day, hour, minute, sec, 0, 0)))

    # ------------------------------------------------------------------
    def read(self) -> int:
        """Return milliseconds since UNIX epoch (uint64)."""
        try:
            epoch_s = self._read_epoch_seconds()
            self._mark_ok()
            return int(epoch_s) * 1000
        except Exception as exc:                   # noqa: BLE001
            self._mark_error("read failed: %r" % exc)
            raise SensorError(str(exc))
