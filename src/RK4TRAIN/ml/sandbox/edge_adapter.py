"""Contract adapter between EDGE-AURA-MFP sensor packets and the sandbox loop.

This module is the single place where the EDGE wire format
(``PINN_SENSOR_PACKET_SCHEMA``, defined in
``src/DAQ4MPC/workstation/packet_schema.py`` / ``src/DAQ4MPC/workstation/packet_builder.py``) is
translated into the flat conditions dict `decision_server.py` uses, and
back the other way: building a pose command in the schema an EDGE-side
actuator would consume (currently unused -- see `pose_to_edge_command`'s
own docstring, manual actuation is confirmed).

Why this exists
----------------
EDGE and the workstation speak different vocabularies:

* EDGE speaks ``G_poa`` / ``T_amb`` / ``WS`` / ``CC`` (a flat, wire-cheap
  sensor packet with no humidity, pressure, wind direction, or elevation
  fields — the physical BOM has no sensors for those). ``G_poa``/``CC``
  are ALWAYS null on the wire -- irradiance and cloud_cover are manual,
  operator-supplied session constants now (camera dropped, PSO
  irradiance estimator cut, see checklist), not sensed at all.
* This module's output is a flat dict (``ambient_c``, ``wind_mps``,
  ``alt``, ``wind_dir``, ``humidity``, ``pressure``, plus derived time
  fields) that `decision_server.py`'s `_edge_conditions_to_pinn_groups()`
  translates further into the grouped weather/location dicts
  `ClosedLoopRuntime` actually expects. irradiance/cloud_cover are
  DELIBERATELY NOT part of this output at all -- see
  `edge_packet_to_conditions`'s docstring for why, this was a real bug
  fixed 2026-08-22.
* The physical panel joint (see ``hardware/AURA_MFP_panel_joint.scad``) is a
  4-DoF pitch/yaw/roll/z gimbal-on-a-post -- the pose schema
  `pose_to_edge_command` builds, when/if anything calls it again.

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
}

@dataclass
class ConditionsResult:
    """Flat conditions dict extracted from an EDGE packet, plus provenance.
    Feeds into decision_server.py's _edge_conditions_to_pinn_groups()."""

    conditions: dict[str, Any]
    field_sources: dict[str, str] = field(default_factory=dict)
    """Maps each returned field name to 'edge' | 'default' | 'derived'."""
    degraded: bool = False
    """True if fault_flags was non-zero on this packet. NOT set for
    G_poa/CC being null -- that's the permanent expected state (manual
    irradiance/cloud_cover entry), not a fault; see module docstring."""
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
    """Convert one ``PINN_SENSOR_PACKET_SCHEMA`` dict into a flat conditions
    dict of everything EDGE can genuinely report or a station default
    covers. irradiance/cloud_cover deliberately NOT included -- see
    module docstring, they're operator-supplied session constants, not
    derived from a packet.

    Parameters
    ----------
    packet:
        A validated EDGE sensor packet (see ``workstation.packet_builder.build_sensor_packet``).
    station_overrides:
        Per-deployment values for fields EDGE cannot measure (``alt``,
        ``wind_dir``, ``humidity``, ``pressure``).
        Merged over :data:`STATION_DEFAULTS`.

    Returns
    -------
    ConditionsResult
        ``conditions`` is a flat dict of sensed/defaulted/derived fields
        (NOT irradiance/cloud_cover -- see above).
        ``field_sources`` records, per field, whether the value came from
        the EDGE packet, a station default, or was derived (time fields).
        ``degraded`` is set only when ``fault_flags`` was non-zero --
        callers (typically :mod:`sandbox.decision_server`) should still
        run a decision cycle (graceful degradation), but should flag the
        cycle as untrustworthy in logs/UI rather than silently trusting
        a filled-in default.
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

    # G_poa and CC are ALWAYS null on the wire now (irradiance/cloud_cover
    # are manual, operator-supplied session constants -- camera dropped,
    # PSO irradiance estimator cut, see checklist). They are deliberately
    # NOT included in the returned conditions dict at all -- a previous
    # version derived them here (falling back to 0.0 on every packet,
    # flagged "degraded"/"treat as night") which meant every single
    # post-calibration packet silently overwrote the operator's real
    # manually-entered irradiance back to 0.0 via inject_conditions()'s
    # dict.update() semantics. Found by tracing the actual data flow, not
    # by inspection. This function now only reports fields EDGE can
    # genuinely sense (or has a real station default for) -- irradiance/
    # cloud_cover stay exactly what calibrate()/calibrate_interactive()
    # set them to, untouched by anything derived from a packet.

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
    """Build a pitch/yaw/roll/z pose command in the schema an EDGE-side
    actuator would consume.

    CURRENTLY UNUSED (2026-08-22): confirmed manual actuation this session
    -- decision_server.py's handle_packet() no longer calls this at all,
    it logs the recommendation for the operator instead of building a
    command (see that file's comment right where this used to be called).
    ``pi.actuator_stub.ActuatorStub``, the consumer this was originally
    built for, was deleted along with the rest of the dead Pi-hosted
    architecture. Kept, not deleted, in case an automated actuator gets
    added later and this pose schema is still the right shape for it --
    but as of now nothing in the live path calls this function.
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
