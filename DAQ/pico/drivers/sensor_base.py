"""pico/drivers/sensor_base.py – Abstract base class for all sensor drivers.

Every driver must subclass :class:`SensorBase` and implement :meth:`read`.
The uniform interface is::

    init(self, **pins)                  # constructor – wire up hardware
    read(self)    -> int | tuple | dict # returns raw counts; raises SensorError on failure
    calibrate(self, raw)                # raw → physical unit; default passthrough
    last_error(self) -> str | None      # most recent error message

Swapping hardware = swapping one driver file.

MicroPython does not ship the standard-library ``abc`` module, so the ABC
behaviour is emulated with a simple ``NotImplementedError`` guard.
"""


class SensorError(Exception):
    """Raised by a driver's :meth:`read` on transient hardware failure."""


class SensorBase:
    """Abstract base for a sensor driver.

    Attributes
    ----------
    sensor_id : int
        Numeric identifier (0=pyranometer, 1..4=thermocouple, 5=anemometer, 6=rtc).
    name : str
        Human-readable sensor name used in log messages.
    """

    sensor_id: int = 0
    name: str = "base"

    def __init__(self, **pins):
        self._pins = pins
        self._last_error: "str | None" = None
        self._consecutive_errors: int = 0

    # ------------------------------------------------------------------
    def read(self):
        """Sample the sensor and return raw counts.

        Returns
        -------
        int | tuple | dict
            Raw counts / registers (untransformed). Subclasses define shape.

        Raises
        ------
        SensorError
            If a hardware communication error occurs.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement read()")

    # ------------------------------------------------------------------
    def calibrate(self, raw):
        """Convert raw counts to a physical unit (default: identity).

        ``pi/calibration.py`` typically handles unit conversion; drivers
        may override this for trivial on-device scaling if desired.
        """
        return raw

    # ------------------------------------------------------------------
    def last_error(self) -> "str | None":
        """Return the most recent error message, or None if healthy."""
        return self._last_error

    # ------------------------------------------------------------------
    def _mark_ok(self) -> None:
        self._last_error = None
        self._consecutive_errors = 0

    def _mark_error(self, msg: str) -> None:
        self._last_error = msg
        self._consecutive_errors += 1

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.sensor_id} name={self.name!r}>"
