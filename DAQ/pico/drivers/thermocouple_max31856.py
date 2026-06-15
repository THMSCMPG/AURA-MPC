"""pico/drivers/thermocouple_max31856.py – MAX31856 thermocouple amplifier driver.

Supports up to four MAX31856 devices sharing a single SPI bus, each
with its own chip-select pin.  Returns raw linearised counts for each
channel; physical °C conversion is handled on the Pi side by
``pi/calibration.py``.

Datasheet: https://datasheets.maximintegrated.com/en/ds/MAX31856.pdf

Wiring (SPI0, four channels)
----------------------------
    MAX31856 SCK  → GP2
    MAX31856 SDI  → GP3  (MOSI)
    MAX31856 SDO  → GP4  (MISO)
    MAX31856 CS0..CS3 → GP5, GP6, GP7, GP8  (active-low, one per device)
    MAX31856 VCC  → 3.3 V
    MAX31856 GND  → GND

The driver is written to boot cleanly on a bare Pico with no peripherals
attached: any SPI failure is caught and a dummy value is emitted so the
rest of the firmware can continue to produce frames for dev-machine
testing.
"""

try:  # pragma: no cover – only present on MicroPython
    from machine import Pin  # type: ignore
    _MICROPYTHON = True
except ImportError:
    _MICROPYTHON = False

from .sensor_base import SensorBase, SensorError

# ── MAX31856 register addresses ────────────────────────────────────────────
_REG_CR0          = 0x00
_REG_CR1          = 0x01
_REG_MASK         = 0x02
_REG_LTCBH        = 0x0C         # Linearised TC temperature (24-bit, MSB)

# ── Thermocouple type codes (CR1[3:0]) ─────────────────────────────────────
TC_TYPE_K = 0x03
TC_TYPE_J = 0x00
TC_TYPE_T = 0x07
TC_TYPE_N = 0x04

# ── CR0 bits ───────────────────────────────────────────────────────────────
_CR0_CMODE = 0x80  # auto-conversion mode (50/60 Hz)


class ThermocoupleMAX31856(SensorBase):
    """Driver for a bank of 1–4 MAX31856 SPI thermocouple amplifiers.

    Parameters
    ----------
    spi : machine.SPI | None
        Initialised SPI bus. Pass ``None`` for bare-Pico dev mode.
    cs_pins : sequence of int
        GPIO pin numbers for the chip-select lines (one per channel).
    tc_type : int
        Thermocouple type constant applied to every channel (default K).
    """

    sensor_id = 0x02
    name = "thermocouple_max31856"

    def __init__(self, spi=None, cs_pins=(5, 6, 7, 8), tc_type: int = TC_TYPE_K) -> None:
        super().__init__(spi=spi, cs_pins=cs_pins, tc_type=tc_type)
        self._spi = spi
        self._tc_type = tc_type
        self._channels = tuple(cs_pins)
        self._cs = []
        if _MICROPYTHON and spi is not None:
            for pin_no in self._channels:
                self._cs.append(Pin(pin_no, Pin.OUT, value=1))
            try:
                for ch in range(len(self._channels)):
                    self._configure(ch)
            except Exception as exc:  # pragma: no cover – hardware path
                self._mark_error("configure failed: %r" % exc)

    # ------------------------------------------------------------------
    def _select(self, ch: int, state: int) -> None:
        if self._cs:
            self._cs[ch](state)

    def _write_reg(self, ch: int, reg: int, value: int) -> None:
        if self._spi is None:
            return
        buf = bytes([reg | 0x80, value])
        self._select(ch, 0)
        self._spi.write(buf)
        self._select(ch, 1)

    def _read_reg(self, ch: int, reg: int, length: int = 1) -> bytes:
        if self._spi is None:
            return bytes(length)
        tx = bytes([reg & 0x7F]) + bytes(length)
        rx = bytearray(1 + length)
        self._select(ch, 0)
        self._spi.write_readinto(tx, rx)
        self._select(ch, 1)
        return bytes(rx[1:])

    def _configure(self, ch: int) -> None:
        self._write_reg(ch, _REG_CR0, _CR0_CMODE)
        self._write_reg(ch, _REG_CR1, self._tc_type & 0x0F)

    # ------------------------------------------------------------------
    def read(self):
        """Return a list of four raw TC counts (int32 per channel).

        On a bare Pico with no MAX31856 attached, returns a list of
        plausible fake values so the firmware can still boot and emit
        frames for development.

        Raises
        ------
        SensorError
            On SPI communication failure for every configured channel.
        """
        values = []
        failures = 0
        for ch in range(4):
            if ch >= len(self._channels) or self._spi is None:
                # Dev fallback – emit a plausible ~25 °C value.
                values.append(int(25.0 / 0.0078125))
                continue
            try:
                raw = self._read_reg(ch, _REG_LTCBH, 3)
                tc_int = (raw[0] << 16 | raw[1] << 8 | raw[2]) >> 5
                if tc_int & 0x40000:                 # 19-bit sign
                    tc_int -= 0x80000
                values.append(tc_int)
            except Exception as exc:                  # noqa: BLE001
                failures += 1
                values.append(-(1 << 31))             # INT32_MIN = sensor fault
                self._mark_error("ch%d: %r" % (ch, exc))

        if failures == 4:
            raise SensorError("all TC channels failed: %s" % self._last_error)
        if failures == 0:
            self._mark_ok()
        return values
