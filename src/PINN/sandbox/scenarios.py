"""Named environmental condition presets for the closed-loop demo.

These are convenience presets for :meth:`PanelEnv.reset` /
:class:`sandbox.runtime.ClosedLoopRuntime.reset` — used by the MATLAB
simulator's scenario dropdown and by anyone scripting a demo without a
live EDGE feed. Each preset is a plain dict matching
``EpisodeConditions.from_mapping``'s expected keys.

These are deliberately named after physically distinct regimes so a demo
audience can connect a scenario name to what they should expect to see
in the PINN-vs-RK4TRAN drift plot and the decision trace, and so they
loosely mirror the EDGE-side packet scenarios in
``pi/scripts/simulate.py`` (``partly_cloudy``, ``fault_inject``, ...)
without being required to match one-for-one — the EDGE scenarios test
wire-level edge cases (nulls, faults), these test control-relevant
physical regimes.
"""

from __future__ import annotations

from typing import Any

SCENARIOS: dict[str, dict[str, Any]] = {
    "clear_sky_noon": {
        "description": "Clear midday sun, light breeze — steady-state baseline.",
        "lat": 36.17, "lon": -86.78, "alt": 180.0,
        "day_of_year": 172, "month": 6, "year": 2026, "hour": 12.0, "minute": 0.0,
        "ambient_c": 29.0, "wind_mps": 2.5, "wind_dir": 200.0,
        "humidity": 0.35, "irradiance": 1020.0, "cloud_cover": 0.02,
        "pressure": 101100.0,
    },
    "cloud_ramp": {
        "description": "Fast-moving partial cloud deck — high irradiance rate-of-change.",
        "lat": 36.17, "lon": -86.78, "alt": 180.0,
        "day_of_year": 172, "month": 6, "year": 2026, "hour": 14.0, "minute": 0.0,
        "ambient_c": 27.0, "wind_mps": 6.0, "wind_dir": 240.0,
        "humidity": 0.55, "irradiance": 640.0, "cloud_cover": 0.55,
        "pressure": 100800.0,
    },
    "high_wind_transient": {
        "description": "Gusty conditions that stress the mechanical rate limits and cooling.",
        "lat": 40.0, "lon": -105.3, "alt": 1600.0,
        "day_of_year": 100, "month": 4, "year": 2026, "hour": 15.0, "minute": 0.0,
        "ambient_c": 18.0, "wind_mps": 16.0, "wind_dir": 270.0,
        "humidity": 0.20, "irradiance": 780.0, "cloud_cover": 0.15,
        "pressure": 84500.0,
    },
    "low_light_dawn": {
        "description": "Low sun angle shortly after sunrise — tests glancing incidence.",
        "lat": 36.17, "lon": -86.78, "alt": 180.0,
        "day_of_year": 172, "month": 6, "year": 2026, "hour": 6.5, "minute": 0.0,
        "ambient_c": 19.0, "wind_mps": 1.0, "wind_dir": 90.0,
        "humidity": 0.70, "irradiance": 210.0, "cloud_cover": 0.10,
        "pressure": 101300.0,
    },
    "hot_desert_noon": {
        "description": "High ambient temperature stresses the thermal derating model.",
        "lat": 33.4, "lon": -112.1, "alt": 330.0,
        "day_of_year": 200, "month": 7, "year": 2026, "hour": 13.0, "minute": 0.0,
        "ambient_c": 44.0, "wind_mps": 3.0, "wind_dir": 180.0,
        "humidity": 0.08, "irradiance": 1080.0, "cloud_cover": 0.0,
        "pressure": 100500.0,
    },
    "night_standby": {
        "description": "No usable irradiance — panel should settle to a park pose.",
        "lat": 36.17, "lon": -86.78, "alt": 180.0,
        "day_of_year": 172, "month": 6, "year": 2026, "hour": 23.0, "minute": 0.0,
        "ambient_c": 16.0, "wind_mps": 1.5, "wind_dir": 150.0,
        "humidity": 0.60, "irradiance": 0.0, "cloud_cover": 0.30,
        "pressure": 101400.0,
    },
}


def list_scenarios() -> list[str]:
    """Return scenario names in a stable, demo-friendly order."""
    return list(SCENARIOS.keys())


def get_scenario(name: str) -> dict[str, Any]:
    """Return a copy of the named preset's conditions (description included)."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}; available: {list_scenarios()}")
    return dict(SCENARIOS[name])
