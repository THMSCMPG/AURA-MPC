"""Sensor-driver interfaces for PINN-AURA-MFP (Batch E — Day 10).

A *sensor driver* is anything that yields :class:`SensorPacket` objects
on demand. Real hardware drivers live outside this repo; here we ship
:class:`FakeSensorDriver`, a deterministic synthetic generator suitable
for development, CI, and demos.

A sensor driver is expected to implement a tiny protocol:

* ``poll() -> SensorPacket`` — produce the next reading.
* Optional ``close()`` — release any underlying resources.

Real drivers should be non-blocking on the hot path; prefer a background
thread that pushes into a :class:`queue.Queue` over doing blocking reads
inside ``poll()``. See ``docs/ORCHESTRATOR.md`` for the extension guide.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Protocol

import numpy as np

from .data import SensorPacket


class SensorDriver(Protocol):
    """Structural protocol for sensor drivers consumed by the CLI."""

    def poll(self) -> SensorPacket:
        """Return the next :class:`SensorPacket`."""
        ...


class FakeSensorDriver:
    """Deterministic synthetic :class:`SensorPacket` generator.

    The driver advances a virtual clock by :attr:`cadence_s` per call to
    :meth:`poll` and emits physically plausible readings:

    * ``G_poa`` follows a clipped cosine diurnal (max near solar noon).
    * ``T_amb`` follows a 3-hour-lagged sinusoid.
    * ``WS`` is a small log-normal draw.
    * ``CC`` is a slow random walk in ``[0, 1]``.
    * ``lat, lon`` are fixed at the constructor values.

    A local :class:`numpy.random.Generator` is used so the driver never
    touches global RNG state.
    """

    def __init__(
        self,
        *,
        cadence_s: float = 3.0,
        seed: int = 0,
        start_time: datetime | None = None,
        lat: float = 37.5,
        lon: float = -122.0,
    ) -> None:
        """Configure the driver.

        Args:
            cadence_s: Virtual seconds advanced per :meth:`poll` call.
            seed: Local RNG seed.
            start_time: Initial virtual wall-clock. Defaults to
                ``datetime.now()`` at construction time.
            lat: Fixed latitude (deg).
            lon: Fixed longitude (deg).

        Raises:
            ValueError: If ``cadence_s`` is not positive.
        """
        if cadence_s <= 0:
            raise ValueError(f"cadence_s must be > 0, got {cadence_s}")
        self.cadence_s = float(cadence_s)
        self._rng = np.random.default_rng(int(seed))
        self._now = start_time if start_time is not None else datetime.now()
        self._lat = float(lat)
        self._lon = float(lon)
        # CC random walk state.
        self._cc = float(self._rng.uniform(0.1, 0.6))

    def poll(self) -> SensorPacket:
        """Emit the next :class:`SensorPacket`."""
        self._now = self._now + timedelta(seconds=self.cadence_s)
        t_s = float(
            self._now.hour * 3600
            + self._now.minute * 60
            + self._now.second
            + self._now.microsecond / 1_000_000.0
        )
        t_h = t_s / 3600.0

        diurnal = max(0.0, math.cos((t_h - 12.0) * math.pi / 12.0))
        g_poa_raw = 950.0 * diurnal + float(self._rng.normal(0.0, 20.0))
        g_poa = float(np.clip(g_poa_raw, 0.0, 1400.0))

        t_amb_raw = (
            22.0
            + 8.0 * math.sin((t_h - 15.0) * math.pi / 12.0)
            + float(self._rng.normal(0.0, 0.5))
        )
        t_amb = float(np.clip(t_amb_raw, -40.0, 70.0))

        ws = float(np.clip(self._rng.lognormal(mean=1.0, sigma=0.3), 0.0, 60.0))

        # Slow random walk for cloud cover.
        self._cc = float(
            np.clip(self._cc + self._rng.normal(0.0, 0.02), 0.0, 1.0)
        )

        return SensorPacket(
            timestamp=self._now,
            t_s=t_s,
            G_poa=g_poa,
            T_amb=t_amb,
            WS=ws,
            CC=self._cc,
            lat=self._lat,
            lon=self._lon,
        )

    def close(self) -> None:
        """No-op; retained for interface parity with real drivers."""
        return None
