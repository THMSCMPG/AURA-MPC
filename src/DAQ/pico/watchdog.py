"""pico/watchdog.py – Thin wrapper around the RP2040 hardware watchdog timer.

Usage
-----
    from watchdog import Watchdog
    wdt = Watchdog(timeout_ms=8000)
    wdt.start()
    while True:
        do_work()
        wdt.feed()   # must be called within timeout_ms or the MCU resets
"""

from machine import WDT


class Watchdog:
    """Hardware watchdog timer wrapper.

    Parameters
    ----------
    timeout_ms:
        Watchdog timeout in milliseconds.  The RP2040 supports a maximum of
        8 388 ms.  The MCU will hard-reset if :meth:`feed` is not called
        within this window after the last feed (or after :meth:`start`).
    """

    def __init__(self, timeout_ms: int = 8000) -> None:
        self._timeout_ms = timeout_ms
        self._wdt: WDT | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Arm the watchdog.  Cannot be disabled once started."""
        self._wdt = WDT(id=0, timeout=self._timeout_ms)

    # ------------------------------------------------------------------
    def feed(self) -> None:
        """Reset the watchdog countdown.  Call this regularly in the main loop."""
        if self._wdt is not None:
            self._wdt.feed()
