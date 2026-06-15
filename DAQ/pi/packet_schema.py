"""pi/packet_schema.py – PINN_SENSOR_PACKET_SCHEMA and FIELD_RANGES for the gateway.

Defines the Python-type schema for packets received from the Pi Pico C firmware
over UART and the physical range bounds used during validation.

The C firmware emits a JSON line per sample matching this schema.  The
``timestamp`` field emitted by the firmware is normalised to ``timestamp_iso``
by :class:`pi.gateway.PicoGateway` before validation.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Python-type schema
# ---------------------------------------------------------------------------
# Each value is a type or tuple-of-types accepted by isinstance().
# (float, type(None)) means the field may be a float or JSON null.

PINN_SENSOR_PACKET_SCHEMA: dict = {
    "timestamp_iso":   str,
    "G_poa":           float,
    "T_amb":           float,
    "WS":              float,
    "CC":              (float, type(None)),
    "lat":             float,
    "lon":             float,
    "azimuth":         (float, type(None)),
    "tilt":            (float, type(None)),
    "height":          (float, type(None)),
    "fault_flags":     int,
    "sky_image_path":  (str, type(None)),
    "pose":            (dict, type(None)),
    "edge_version":    str,
}

# Required fields that must be present and non-None in every valid packet.
REQUIRED_FIELDS: tuple[str, ...] = (
    "timestamp_iso",
    "G_poa",
    "T_amb",
    "WS",
    "fault_flags",
    "edge_version",
)

# ---------------------------------------------------------------------------
# Physical range bounds (inclusive) for sensor readings
# ---------------------------------------------------------------------------

FIELD_RANGES: dict[str, tuple[float, float]] = {
    "G_poa": (0.0,   1400.0),
    "T_amb": (-30.0,   70.0),
    "WS":    (0.0,     50.0),
}
