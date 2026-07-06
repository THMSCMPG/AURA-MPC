"""pi/gateway.py – Edge gateway: UART packet ingestion, validation, and buffering.

Reads newline-delimited JSON ``PINN_SENSOR_PACKET_SCHEMA`` packets from a
serial-like source (real ``serial.Serial`` or any object with a
``readline() -> bytes`` method), validates them, and accumulates them into a
window buffer.

Validation rules
----------------
* ``fault_flags`` must be 0 — any non-zero value indicates a sensor fault.
* ``G_poa`` must be ≤ :data:`G_POA_MAX` (1400 W/m²) — values above this
  indicate a hardware or calibration fault.
* The JSON must be parse-able and contain all required schema fields.

Window flushing
---------------
A *window* is considered *full* when it contains :data:`WINDOW_FULL` packets.
If the serial source runs dry before the window fills, the gateway will flush
anyway provided at least :data:`WINDOW_MIN` packets have accumulated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pi.packet_builder import validate_sensor_packet

log = logging.getLogger("edge-aura.gateway")

# ── Tunables ──────────────────────────────────────────────────────────────────

G_POA_MAX: float = 1400.0
"""Maximum accepted plane-of-array irradiance (W/m²)."""

WINDOW_FULL: int = 60
"""Target window size — flush when this many valid packets accumulate."""

WINDOW_MIN: int = 10
"""Minimum window size — flush is allowed once this threshold is reached."""


# ── Exception ─────────────────────────────────────────────────────────────────

class PacketValidationError(ValueError):
    """Raised when a packet fails gateway validation."""


# ── Validation ────────────────────────────────────────────────────────────────

def validate_gateway_packet(packet: dict[str, Any]) -> None:
    """Validate *packet* for gateway ingestion.

    Runs the schema validation from :mod:`pi.packet_builder` and then applies
    gateway-specific range checks.

    Raises
    ------
    PacketValidationError
        If the packet fails any validation rule.
    """
    try:
        validate_sensor_packet(packet)
    except ValueError as exc:
        raise PacketValidationError(str(exc)) from exc

    ff = packet.get("fault_flags", 0)
    if ff != 0:
        raise PacketValidationError(f"non-zero fault_flags: {ff:#06x}")

    g_poa = packet.get("G_poa")
    if g_poa is not None and g_poa > G_POA_MAX:
        raise PacketValidationError(
            f"G_poa {g_poa} W/m² exceeds maximum {G_POA_MAX} W/m²"
        )


# ── Gateway ───────────────────────────────────────────────────────────────────

class Gateway:
    """Read, validate, and buffer PINN sensor packets from a serial-like source.

    Parameters
    ----------
    serial:
        Any object that exposes ``readline() -> bytes`` and ``close()``.
        Typically a ``serial.Serial`` instance or a :class:`MockSerial`.
    window_size:
        Target number of valid packets per window (default: :data:`WINDOW_FULL`).
    min_flush:
        Minimum number of valid packets required for a partial flush
        (default: :data:`WINDOW_MIN`).

    Examples
    --------
    ::

        gw = Gateway(serial_port, window_size=60, min_flush=10)
        packets = gw.ingest()          # blocks until window full or EOF
        if len(packets) >= gw.min_flush:
            csv_data = packets_to_csv(packets)
            metrics  = call_simv2b_runner(csv_data)
    """

    def __init__(
        self,
        serial: Any,
        *,
        window_size: int = WINDOW_FULL,
        min_flush: int = WINDOW_MIN,
    ) -> None:
        self._serial = serial
        self.window_size = window_size
        self.min_flush = min_flush
        self._buffer: list[dict] = []
        self._rejected: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def rejected(self) -> int:
        """Number of packets rejected since construction."""
        return self._rejected

    def ingest(self) -> list[dict]:
        """Read packets from the serial source until the window is full or EOF.

        Each line is decoded as UTF-8 JSON, validated, and either appended to
        the internal buffer or counted as a rejected packet.

        Returns
        -------
        list[dict]
            A *copy* of the buffer at the time the call returns.  The buffer
            itself is reset so that a subsequent call to ``ingest()`` starts
            a fresh window.
        """
        self._buffer = []
        while len(self._buffer) < self.window_size:
            raw = self._serial.readline()
            if not raw:
                break
            try:
                packet = json.loads(raw.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("gateway: malformed line — %s", exc)
                self._rejected += 1
                continue
            try:
                validate_gateway_packet(packet)
            except PacketValidationError as exc:
                log.debug("gateway: rejected packet — %s", exc)
                self._rejected += 1
                continue
            self._buffer.append(packet)

        return list(self._buffer)

    def close(self) -> None:
        """Close the underlying serial connection."""
        try:
            self._serial.close()
        except Exception:  # noqa: BLE001
            pass
