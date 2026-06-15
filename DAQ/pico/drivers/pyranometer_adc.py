"""pico/drivers/pyranometer_adc.py – ADC-based pyranometer driver.

Reads the analogue voltage produced by a Li-Cor LI-200R (or compatible)
pyranometer connected to one of the RP2040's ADC inputs.  The raw 16-bit
ADC count is returned; physical conversion (W/m²) is handled on the Pi
side by ``pi/calibration.py``.

On a bare Pico with no sensor attached, the driver emits a slow sine
wave so the rest of the firmware boots and produces frames for dev
testing.

Wiring
------
    Pyranometer (+) → GP26 (ADC0) via signal-conditioning resistor
    Pyranometer (−) → GND
"""

try:  # pragma: no cover
    from machine import ADC, Pin  # type: ignore
    _MICROPYTHON = True
except ImportError:
    _MICROPYTHON = False
    import math
    import time as _time

from .sensor_base import SensorBase, SensorError


class PyranometerADC(SensorBase):
    """Irradiance sensor using the RP2040 on-chip ADC.

    Parameters
    ----------
    adc_pin : int
        GPIO pin (must be ADC-capable: 26, 27, or 28 on the Pico).
    oversampling : int
        Number of readings averaged per :meth:`read` (>=1).
    """

    sensor_id = 0x01
    name = "pyranometer"

    def __init__(self, adc_pin: int = 26, oversampling: int = 8) -> None:
        super().__init__(adc_pin=adc_pin, oversampling=oversampling)
        self._adc = None
        self._oversampling = max(1, oversampling)
        if _MICROPYTHON:
            try:
                self._adc = ADC(Pin(adc_pin))
            except Exception as exc:        # pragma: no cover
                self._mark_error("ADC init failed: %r" % exc)

    # ------------------------------------------------------------------
    def read(self) -> int:
        """Return the averaged 16-bit ADC count.

        Returns
        -------
        int
            Raw counts in the range 0–65535.

        Raises
        ------
        SensorError
            On ADC read failure.
        """
        try:
            if self._adc is None:
                # Dev fallback – emit a midday-ish value.
                if not _MICROPYTHON:
                    t = _time.monotonic()
                    value = int(32768 + 16000 * math.sin(t / 10.0))
                    value = max(0, min(65535, value))
                else:
                    value = 32768
                self._mark_ok()
                return value
            total = 0
            for _ in range(self._oversampling):
                total += self._adc.read_u16()
            value = total // self._oversampling
            self._mark_ok()
            return value
        except Exception as exc:             # noqa: BLE001
            self._mark_error("read failed: %r" % exc)
            raise SensorError(str(exc))
