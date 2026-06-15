"""pi/packet_builder.py – Assemble SensorPacket JSON matching the PINN contract.

The PINN-AURA-MFP orchestrator consumes ``SensorPacket`` JSON objects
one per sample interval.  The schema is intentionally flat and stable
across protocol versions – new fields append at the bottom.

SensorPacket schema (v1.0)
--------------------------

    {
      "schema_version":      "1.0",
      "timestamp_utc":       "<ISO-8601 UTC>",
      "timestamp_ms":        <uint64>,          // echo of sensor clock
      "irradiance_w_m2":     <float | null>,    // null ⇒ sensor fault
      "thermocouples_c":     [<float | null> × 4],
      "wind_speed_m_s":      <float | null>,
      "wind_direction_deg":  <float | null>,
      "fault_flags":         <uint16>,          // bitmask (see below)
      "image_path":          <str | null>       // populated by Edge-Batch B
    }

Fault-flag bitmask
------------------

    0x0001  pyranometer        0x0010  thermocouple 3
    0x0002  thermocouple 0     0x0020  anemometer
    0x0004  thermocouple 1     0x0040  RTC
    0x0008  thermocouple 2     0x8000  persistent-fault indicator

The JSON Schema (draft-2020-12) that validates these packets is exposed
as :data:`SENSOR_PACKET_SCHEMA` and written to
``docs/sensor_packet.schema.json`` during ``pi.daemon --help``.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


# ══════════════════════════════════════════════════════════════════════
# DEPRECATED — retained only for backwards compatibility with pre-Batch-A
# callers.  The PINN-AURA-MFP orchestrator consumes ``build_sensor_packet()``
# / ``PINN_SENSOR_PACKET_SCHEMA`` ONLY.  New code must not call
# ``build_packet()``.
# ══════════════════════════════════════════════════════════════════════

SENSOR_PACKET_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id":     "https://aura-mfp.example.com/sensor_packet.schema.json",
    "title":   "SensorPacket",
    "type":    "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "timestamp_utc", "timestamp_ms",
        "irradiance_w_m2", "thermocouples_c",
        "wind_speed_m_s", "wind_direction_deg",
        "fault_flags", "image_path",
    ],
    "properties": {
        "schema_version":     {"const": SCHEMA_VERSION},
        "timestamp_utc":      {"type": "string", "format": "date-time"},
        "timestamp_ms":       {"type": "integer", "minimum": 0},
        "irradiance_w_m2":    {"type": ["number", "null"]},
        "thermocouples_c": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": ["number", "null"]},
        },
        "wind_speed_m_s":     {"type": ["number", "null"], "minimum": 0},
        "wind_direction_deg": {
            "oneOf": [
                {"type": "null"},
                {"type": "number", "minimum": 0, "maximum": 360},
            ],
        },
        "fault_flags":        {"type": "integer", "minimum": 0, "maximum": 0xFFFF},
        "image_path":         {"type": ["string", "null"]},
    },
}


def build_packet(
    *,
    timestamp_ms: int,
    irradiance_w_m2: Optional[float],
    thermocouples_c,
    wind_speed_m_s: Optional[float],
    wind_direction_deg: Optional[float],
    fault_flags: int,
    image_path: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Return a SensorPacket dict.

    .. deprecated::
        Use :func:`build_sensor_packet` instead.  This function is retained
        only for backwards compatibility with pre-Batch-A callers.

    ``now_utc`` is injectable for deterministic tests and replay; when
    ``None`` the current wall-clock UTC time is used.
    """
    warnings.warn(
        "build_packet() is deprecated — use build_sensor_packet() instead. "
        "The PINN-AURA-MFP orchestrator consumes build_sensor_packet() / "
        "PINN_SENSOR_PACKET_SCHEMA ONLY.",
        DeprecationWarning,
        stacklevel=2,
    )
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    return {
        "schema_version":     SCHEMA_VERSION,
        "timestamp_utc":      now_utc.isoformat(),
        "timestamp_ms":       int(timestamp_ms),
        "irradiance_w_m2":    None if irradiance_w_m2 is None else float(irradiance_w_m2),
        "thermocouples_c":    [None if v is None else float(v) for v in thermocouples_c],
        "wind_speed_m_s":     None if wind_speed_m_s is None else float(wind_speed_m_s),
        "wind_direction_deg": None if wind_direction_deg is None else float(wind_direction_deg),
        "fault_flags":        int(fault_flags) & 0xFFFF,
        "image_path":         image_path,
    }


def validate_packet(packet: dict) -> None:
    """Validate *packet* against :data:`SENSOR_PACKET_SCHEMA`.

    Uses ``jsonschema`` when available; otherwise falls back to a
    lightweight hand-rolled checker so the daemon still runs without the
    optional dependency.  Raises ``ValueError`` on any violation.
    """
    try:
        import jsonschema  # type: ignore
    except ImportError:
        _fallback_validate(packet)
        return
    try:
        jsonschema.validate(packet, SENSOR_PACKET_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"SensorPacket schema violation: {exc.message}") from exc


# ────────────────────────────── Fallback ──────────────────────────────
def _fallback_validate(p: dict) -> None:
    required = SENSOR_PACKET_SCHEMA["required"]
    for field in required:
        if field not in p:
            raise ValueError(f"missing field: {field}")
    if p["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"bad schema_version: {p['schema_version']!r}")
    if not isinstance(p["timestamp_utc"], str):
        raise ValueError("timestamp_utc must be str")
    if not isinstance(p["timestamp_ms"], int) or p["timestamp_ms"] < 0:
        raise ValueError("timestamp_ms must be non-negative int")
    _num_or_none(p, "irradiance_w_m2")
    tcs = p["thermocouples_c"]
    if not isinstance(tcs, list) or len(tcs) != 4:
        raise ValueError("thermocouples_c must be list of 4")
    for i, v in enumerate(tcs):
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError(f"thermocouples_c[{i}] must be number or null")
    _num_or_none(p, "wind_speed_m_s")
    wd = p["wind_direction_deg"]
    if wd is not None and not (isinstance(wd, (int, float)) and 0 <= wd <= 360):
        raise ValueError("wind_direction_deg out of range")
    ff = p["fault_flags"]
    if not isinstance(ff, int) or not (0 <= ff <= 0xFFFF):
        raise ValueError("fault_flags must be uint16")
    ip = p["image_path"]
    if ip is not None and not isinstance(ip, str):
        raise ValueError("image_path must be string or null")


def _num_or_none(p: dict, key: str) -> None:
    v = p[key]
    if v is not None and not isinstance(v, (int, float)):
        raise ValueError(f"{key} must be number or null")


# ══════════════════════════════════════════════════════════════════════
# PINN-AURA-MFP Batch A SensorPacket (Edge-Batch B)
# ══════════════════════════════════════════════════════════════════════
#
# Schema documented verbatim in the project plan. The orchestrator
# consumes this shape; the edge does NOT embed image bytes (too large)
# — it ships only the path to a JPEG that the orchestrator reads over a
# shared filesystem mount.

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
    """Return a PINN-AURA-MFP Batch A ``SensorPacket`` dict.

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
