"""workstation/packet_builder.py – Assemble SensorPacket JSON matching the PINN contract.

The PINN-AURA-MFP decision layer (``sandbox.decision_server`` as of the
AURA-MPC EDGE/PINN/RK4TRAN wiring pass) consumes ``SensorPacket`` JSON
objects one per sample interval, in the schema below.

.. note:: Cleanup (AURA-MPC wiring pass)
    This file used to also carry a deprecated v1.0 schema
    (``build_packet`` / ``validate_packet`` / ``SENSOR_PACKET_SCHEMA``,
    with ``irradiance_w_m2`` / ``thermocouples_c`` / ``wind_direction_deg``
    fields) alongside the current one below. Every caller in this repo
    had already migrated to ``build_sensor_packet`` /
    ``PINN_SENSOR_PACKET_SCHEMA`` except ``workstation.daemon (deleted)`` and two scripts
    (``workstation.scripts.replay (deleted)``, ``workstation.scripts.health_check``), which were
    fixed in the same pass this file was cleaned up in. The deprecated
    functions have been removed rather than kept around as dead code —
    if you're reading an old integration that still calls
    ``build_packet``/``validate_packet``, it needs to move to the
    schema below.

SensorPacket schema
--------------------

    {
      "timestamp":       "<ISO-8601 UTC, millisecond precision>",
      "t_s":              <float>,               // seconds elapsed (monotonic-ish)
      "G_poa":            <float | null>,         // plane-of-array irradiance, W/m^2
      "T_amb":            <float | null>,         // ambient temperature, deg C
      "WS":               <float | null>,         // wind speed, m/s
      "CC":               <float | null>,         // cloud cover [0,1] — orchestrator-supplied
      "lat":              <float>,                // station latitude
      "lon":              <float>,                // station longitude
      "sky_image_path":   <str | null>,
      "pose":             <object | array | null>,// orchestrator-supplied
      "fault_flags":      <uint16>,                // bitmask (see below)
      "edge_version":     <str>
    }

Fault-flag bitmask
------------------

    0x0001  pyranometer        0x0010  thermocouple 3
    0x0002  thermocouple 0     0x0020  anemometer
    0x0004  thermocouple 1     0x0040  RTC
    0x0008  thermocouple 2     0x8000  persistent-fault indicator

Two fields EDGE's current BOM cannot measure directly are always ``null``
on the edge side and filled in downstream: ``CC`` (cloud cover) and
``pose`` are always produced by the orchestrator/decision layer. Fields
the BOM has no sensor for at all (humidity, pressure, wind direction,
station altitude) aren't part of this wire schema — see
``sandbox.edge_adapter.STATION_DEFAULTS`` for how the decision layer
fills those in.

The JSON Schema (draft-2020-12) that validates these packets is exposed
as :data:`PINN_SENSOR_PACKET_SCHEMA`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

EDGE_VERSION = "v0.1.0"

PINN_SENSOR_PACKET_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id":     "https://aura-mfp.example.com/pinn_sensor_packet.schema.json",
    "title":   "PinnSensorPacket",
    "type":    "object",
    "additionalProperties": False,
    "required": [
        "timestamp", "t_s", "G_poa", "T_amb", "WS", "CC",
        "lat", "lon", "sky_image_path", "pose", "fault_flags",
        "edge_version",
    ],
    "properties": {
        "timestamp":      {"type": "string", "format": "date-time"},
        "t_s":            {"type": "number", "minimum": 0},
        "G_poa":          {"type": ["number", "null"]},
        "T_amb":          {"type": ["number", "null"]},
        "WS":             {"type": ["number", "null"], "minimum": 0},
        "CC":             {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "lat":            {"type": "number", "minimum": -90,  "maximum": 90},
        "lon":            {"type": "number", "minimum": -180, "maximum": 180},
        "sky_image_path": {"type": ["string", "null"]},
        "pose":           {"type": ["object", "array", "null"]},
        "fault_flags":    {"type": "integer", "minimum": 0, "maximum": 0xFFFF},
        "edge_version":   {"type": "string"},
    },
}


def build_sensor_packet(
    *,
    t_s: float,
    G_poa: Optional[float],
    T_amb: Optional[float],
    WS: Optional[float],
    lat: float,
    lon: float,
    fault_flags: int,
    sky_image_path: Optional[str] = None,
    CC: Optional[float] = None,
    pose=None,
    now_utc: Optional[datetime] = None,
    edge_version: str = EDGE_VERSION,
) -> dict[str, Any]:
    """Return a PINN-AURA-MFP ``SensorPacket`` dict.

    Fields match the project plan. ``CC`` (cloud cover) and ``pose``
    are always produced by the orchestrator and hence default to
    ``None`` on the edge side.

    ``now_utc`` is injectable for deterministic tests and replay.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    # Keep millisecond-precision ISO-8601 with trailing Z for consistency.
    ts = now_utc.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") \
        + f"{now_utc.microsecond // 1000:03d}Z"
    return {
        "timestamp":      ts,
        "t_s":            float(t_s),
        "G_poa":          None if G_poa is None else float(G_poa),
        "T_amb":          None if T_amb is None else float(T_amb),
        "WS":             None if WS is None else float(WS),
        "CC":             None if CC is None else float(CC),
        "lat":            float(lat),
        "lon":            float(lon),
        "sky_image_path": sky_image_path,
        "pose":           pose,
        "fault_flags":    int(fault_flags) & 0xFFFF,
        "edge_version":   edge_version,
    }


def validate_sensor_packet(packet: dict) -> None:
    """Validate *packet* against :data:`PINN_SENSOR_PACKET_SCHEMA`.

    Uses ``jsonschema`` when available; otherwise falls back to a
    lightweight hand-rolled checker so the daemon still runs without the
    optional dependency. Raises ``ValueError`` on any violation.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _fallback_validate_pinn(packet)
        return
    try:
        jsonschema.validate(packet, PINN_SENSOR_PACKET_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"PinnSensorPacket schema violation: {exc.message}") from exc


def _num_or_none(p: dict, key: str) -> None:
    v = p[key]
    if v is not None and not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be number or null")


def _fallback_validate_pinn(p: dict) -> None:
    required = PINN_SENSOR_PACKET_SCHEMA["required"]
    for field in required:
        if field not in p:
            raise ValueError(f"missing field: {field}")
    if not isinstance(p["timestamp"], str):
        raise ValueError("timestamp must be str")
    if not isinstance(p["t_s"], (int, float)) or p["t_s"] < 0:
        raise ValueError("t_s must be non-negative number")
    for key in ("G_poa", "T_amb", "WS", "CC"):
        _num_or_none(p, key)
    ws = p["WS"]
    if ws is not None and ws < 0:
        raise ValueError("WS must be non-negative")
    cc = p["CC"]
    if cc is not None and not (0.0 <= cc <= 1.0):
        raise ValueError("CC must be in [0, 1]")
    lat = p["lat"]
    if not isinstance(lat, (int, float)) or not (-90 <= lat <= 90):
        raise ValueError("lat out of range")
    lon = p["lon"]
    if not isinstance(lon, (int, float)) or not (-180 <= lon <= 180):
        raise ValueError("lon out of range")
    sip = p["sky_image_path"]
    if sip is not None and not isinstance(sip, str):
        raise ValueError("sky_image_path must be string or null")
    ff = p["fault_flags"]
    if not isinstance(ff, int) or not (0 <= ff <= 0xFFFF):
        raise ValueError("fault_flags must be uint16")
    if not isinstance(p["edge_version"], str):
        raise ValueError("edge_version must be string")
