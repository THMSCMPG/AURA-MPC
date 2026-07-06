"""pico/drivers/anemometer_rs485.py – RS-485 (Modbus-RTU) anemometer driver.

Queries a generic RS-485 cup anemometer + wind-vane over a Modbus-RTU
link using holding-register reads.  Returns raw wire values (speed × 100
m/s as uint16, direction 0-359° as uint16).

Wiring (UART1, half-duplex via MAX485)
--------------------------------------
    Pico TX → GP8 → MAX485 DI
    Pico RX → GP9 ← MAX485 RO
    Pico GP10 → MAX485 DE + RE  (tie together; HIGH = transmit)

The driver's ``read()`` emits a plausible fake value on a bare Pico so
the firmware still produces frames during development.
"""

try:  # pragma: no cover
    from machine import Pin  # type: ignore
    _MICROPYTHON = True
except ImportError:
    _MICROPYTHON = False
    import math
    import time as _time

from .sensor_base import SensorBase, SensorError

# ── Modbus-RTU constants ───────────────────────────────────────────────────
_FC_READ_HOLDING = 0x03

# Register layout (sensor-specific; adjust to your model)
_REG_SPEED_X100 = 0x0000      # uint16  m/s × 100
_REG_DIR_DEG    = 0x0001      # uint16  0–359


def _crc16_modbus(data: bytes) -> int:
    """Modbus-RTU CRC-16 (polynomial 0xA001, reflected 0x8005)."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


class AnemometerRS485(SensorBase):
    """Modbus-RTU anemometer driver.

    Parameters
    ----------
    uart : machine.UART | None
        Pre-configured half-duplex UART (typically 9600 8N1).  ``None`` in
        dev mode.
    de_pin : int
        GPIO driving the MAX485 DE/RE line.
    slave_addr : int
        Modbus slave address (default 1).
    """

    sensor_id = 0x03
    name = "anemometer_rs485"

    def __init__(self, uart=None, de_pin: int = 10, slave_addr: int = 1) -> None:
        super().__init__(uart=uart, de_pin=de_pin, slave_addr=slave_addr)
        self._uart = uart
        self._slave = slave_addr
        self._de = None
        if _MICROPYTHON and uart is not None:
            try:
                self._de = Pin(de_pin, Pin.OUT, value=0)
            except Exception as exc:      # pragma: no cover
                self._mark_error("DE pin init: %r" % exc)

    # ------------------------------------------------------------------
    def _read_register(self, reg: int) -> int:
        if self._uart is None:
            return 0
        req = bytes([self._slave, _FC_READ_HOLDING, reg >> 8, reg & 0xFF, 0x00, 0x01])
        crc = _crc16_modbus(req)
        frame = req + bytes([crc & 0xFF, crc >> 8])
        if self._de:
            self._de(1)
        self._uart.write(frame)
        if self._de:
            self._de(0)
        resp = self._uart.read(7) or b""
        if len(resp) != 7:
            raise SensorError("short Modbus response (%d bytes)" % len(resp))
        return (resp[3] << 8) | resp[4]

    # ------------------------------------------------------------------
    def read(self) -> tuple:
        """Return ``(speed_x100, direction_deg)`` raw uint16 values."""
        try:
            if self._uart is None:
                # Dev fallback – gentle fake values.
                if not _MICROPYTHON:
                    t = _time.monotonic()
                    speed_x100 = int(250 + 150 * math.sin(t / 5.0))   # ~2.5 m/s
                    direction = int((t * 9) % 360)
                else:
                    speed_x100, direction = 250, 180
                self._mark_ok()
                return (max(0, speed_x100) & 0xFFFF, direction & 0xFFFF)
            speed = self._read_register(_REG_SPEED_X100)
            direction = self._read_register(_REG_DIR_DEG) % 360
            self._mark_ok()
            return (speed & 0xFFFF, direction & 0xFFFF)
        except Exception as exc:                      # noqa: BLE001
            self._mark_error("read failed: %r" % exc)
            raise SensorError(str(exc))
