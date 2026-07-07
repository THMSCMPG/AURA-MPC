"""Contract adapter between EDGE-AURA-MFP sensor packets and the sandbox loop.

This module is the single place where the EDGE wire format
(``PINN_SENSOR_PACKET_SCHEMA``, defined in
``src/DAQ/pi/packet_schema.py`` / ``src/DAQ/pi/packet_builder.py``) is
translated into the sandbox's internal representation
(:class:`sandbox.environment.EpisodeConditions`), and back the other way:
translating a decided pose into a command EDGE can apply to its stepper
motors.

Why this exists
----------------
EDGE, the workstation PINN/RK4TRAN sandbox, and the physical panel joint
each grew their own vocabulary:

* EDGE speaks ``G_poa`` / ``T_amb`` / ``WS`` / ``CC`` (a flat, wire-cheap
  sensor packet with no humidity, pressure, wind direction, or elevation
  fields — the physical BOM has no sensors for those).
* The sandbox speaks ``EpisodeConditions`` (a superset used for RK4TRAN /
  PINN inference: lat, lon, alt, day_of_year, hour, minute, month, year,
  ambient_c, wind_mps, wind_dir, humidity, irradiance, cloud_cover,
  pressure).
* The physical panel joint (see ``AURA_MFP_panel_joint.scad``) is a
  4-DoF pitch/yaw/roll/z gimbal-on-a-post, matching the pose dict already
  used throughout ``environment.py`` — NOT the azimuth/elevation spherical
  pose that ``pi/actuator_stub.py`` was written against.

Every field that EDGE cannot supply is filled from a documented default
or a station-configuration value (see :data:`STATION_DEFAULTS`). Nothing
is silently invented without a name attached to it in ``field_sources``,
so a human auditing a decision trace can tell live-sensor data apart
from a filled-in assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Station-level defaults for fields EDGE's current BOM cannot measure.
# ---------------------------------------------------------------------------
# These are NOT physical constants — they are placeholders that should be
# overridden with real site survey values (site elevation, a fixed/typical
# wind direction, etc.) via `STATION_DEFAULTS.update(...)` at deployment
# time, or per-call via `station_overrides=`.

STATION_DEFAULTS: dict[str, float] = {
    "alt": 100.0,           # metres — EDGE has no altimeter/GPS-Z; survey the site.
    "wind_dir": 180.0,      # degrees — no wind vane on the current BOM.
    "humidity": 0.5,        # fraction — no hygrometer on the current BOM.
    "pressure": 101325.0,   # Pa — no barometer on the current BOM; std atmosphere.
    "cloud_cover": 0.0,     # fraction — used only if CC is null in the packet.
}

@dataclass
class ConditionsResult:
    """Conditions ready for :meth:`PanelEnv.set_conditions`, plus provenance."""

    conditions: dict[str, Any]
    field_sources: dict[str, str] = field(default_factory=dict)
    """Maps each EpisodeConditions field name to 'edge' | 'default' | 'derived'."""
    degraded: bool = False
    """True if fault_flags was non-zero or a required field was null."""
    notes: list[str] = field(default_factory=list)


def _time_components_from_iso(timestamp: str) -> dict[str, float | int]:
    """Parse an EDGE ISO-8601 timestamp into the sandbox's time fields."""
    ts = timestamp
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    day_of_year = int(dt.timetuple().tm_yday)
    return {
        "hour": float(dt.hour),
        "minute": float(dt.minute) + float(dt.second) / 60.0,
        "day_of_year": day_of_year,
        "month": int(dt.month),
        "year": int(dt.year),
    }


def edge_packet_to_conditions(
    packet: dict[str, Any],
    *,
    station_overrides: Optional[dict[str, float]] = None,
) -> ConditionsResult:
    """Convert one ``PINN_SENSOR_PACKET_SCHEMA`` dict into sandbox conditions.

    Parameters
    ----------
    packet:
        A validated EDGE sensor packet (see ``pi.packet_builder.build_sensor_packet``).
    station_overrides:
        Per-deployment values for fields EDGE cannot measure (``alt``,
        ``wind_dir``, ``humidity``, ``pressure``, ``cloud_cover`` fallback).
        Merged over :data:`STATION_DEFAULTS`.

    Returns
    -------
    ConditionsResult
        ``conditions`` is a dict suitable for
        ``EpisodeConditions.from_mapping`` / ``PanelEnv.set_conditions``.
        ``field_sources`` records, per field, whether the value came from
        the EDGE packet, a station default, or was derived (time fields).
        ``degraded`` is set when ``fault_flags`` was non-zero or a
        required numeric field was ``null`` — callers (typically
        :mod:`sandbox.decision_server`) should still run a decision cycle
        (graceful degradation), but should flag the cycle as untrustworthy
        in logs/UI rather than silently trusting a filled-in default.
    """
    defaults = dict(STATION_DEFAULTS)
    if station_overrides:
        defaults.update(station_overrides)

    sources: dict[str, str] = {}
    notes: list[str] = []
    degraded = bool(packet.get("fault_flags", 0))
    if degraded:
        notes.append(f"fault_flags={packet.get('fault_flags'):#06x} on ingest")

    if "lat" in defaults:
        lat = defaults["lat"]
        sources["lat"] = "override"
    else:
        lat = packet.get("lat")
        sources["lat"] = "edge"
    if "lon" in defaults:
        lon = defaults["lon"]
        sources["lon"] = "override"
    else:
        lon = packet.get("lon")
        sources["lon"] = "edge"
    if lat is None or lon is None:
        raise ValueError(
            "EDGE packet missing required lat/lon and no station_overrides "
            "supplied them — cannot locate the panel"
        )

    # T_amb has no station default of its own — a missing ambient reading
    # always counts as degraded and falls back to a neutral 25 C.
    ambient_c = packet.get("T_amb")
    if ambient_c is None:
        ambient_c = 25.0
        sources["ambient_c"] = "default"
        notes.append("ambient_c: T_amb was null in packet; assuming 25 C")
    else:
        sources["ambient_c"] = "edge"

    wind_mps = packet.get("WS")
    if wind_mps is None:
        wind_mps = 0.0
        sources["wind_mps"] = "default"
        notes.append("wind_mps: WS was null in packet; assuming calm (0 m/s)")
    else:
        sources["wind_mps"] = "edge"

    cloud_cover = packet.get("CC")
    if cloud_cover is None:
        cloud_cover = float(defaults["cloud_cover"])
        sources["cloud_cover"] = "default"
    else:
        sources["cloud_cover"] = "edge"

    irradiance = packet.get("G_poa")
    if irradiance is None:
        irradiance = 0.0
        sources["irradiance"] = "default"
        notes.append("irradiance: G_poa was null in packet; assuming 0 W/m^2 (treat as night/fault)")
        degraded = True
    else:
        sources["irradiance"] = "edge"

    time_components = _time_components_from_iso(packet["timestamp"])
    for key in time_components:
        sources[key] = "derived"

    conditions = {
        "lat": float(lat),
        "lon": float(lon),
        "alt": float(defaults["alt"]),
        "ambient_c": float(ambient_c),
        "wind_mps": float(wind_mps),
        "wind_dir": float(defaults["wind_dir"]),
        "humidity": float(defaults["humidity"]),
        "irradiance": float(irradiance),
        "cloud_cover": float(cloud_cover),
        "pressure": float(defaults["pressure"]),
        **time_components,
    }
    sources["alt"] = "default"
    sources["wind_dir"] = "default"
    sources["humidity"] = "default"
    sources["pressure"] = "default"

    return ConditionsResult(
        conditions=conditions,
        field_sources=sources,
        degraded=degraded,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Decision -> EDGE actuator command
# ---------------------------------------------------------------------------

EDGE_COMMAND_SCHEMA_VERSION = "2.0"
"""Bumped from the old azimuth/elevation stub schema (1.0) to the native
pitch/yaw/roll/z gimbal schema that actually matches the panel joint and
everything upstream of it (PanelEnv, RK4TRAN, the PINN pose head)."""


def pose_to_edge_command(
    pose: dict[str, float],
    *,
    decision_reason: str = "",
    discrepancy: Optional[dict[str, float]] = None,
    validation: Optional[dict[str, Any]] = None,
    command_id: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> dict[str, Any]:
    """Build the JSON command EDGE's actuator stub / stepper driver consumes.

    Uses the native ``pitch/yaw/roll/z`` pose representation the rest of
    the stack (PanelEnv, RK4TRANValidator, the PINN pose head) already
    speaks, instead of the azimuth/elevation spherical schema the old
    ``pi.actuator_stub.ActuatorStub`` expected. See
    ``pi/actuator_stub.py`` for the corresponding parsing update.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    command: dict[str, Any] = {
        "schema_version": EDGE_COMMAND_SCHEMA_VERSION,
        "timestamp_utc": now_utc.isoformat(),
        "pose": {
            "pitch_deg": float(pose["pitch"]),
            "yaw_deg": float(pose["yaw"]),
            "roll_deg": float(pose["roll"]),
            "z_m": float(pose["z"]),
        },
        "decision_reason": decision_reason,
    }
    if command_id is not None:
        command["command_id"] = command_id
    if discrepancy is not None:
        command["discrepancy"] = {
            "T_operating_K": float(discrepancy.get("T_operating", 0.0)),
            "eta": float(discrepancy.get("eta", 0.0)),
        }
    if validation is not None:
        command["validation"] = validation
    return command
