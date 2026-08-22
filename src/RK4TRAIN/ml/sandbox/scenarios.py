"""Named environmental condition presets for the closed-loop demo.

These are convenience presets for :meth:`sandbox.runtime.ClosedLoopRuntime.calibrate`
(and a matching :meth:`~sandbox.runtime.ClosedLoopRuntime.recommend` time
context) -- used by the MATLAB simulator's scenario dropdown
(:meth:`sandbox.matlab_bridge.MatlabSimulationBridge.scenario_conditions_json`)
and by anyone scripting a demo without a live EDGE feed.

REWRITTEN 2026-08-22 (was a real, latent bug -- these were still in the
flat EpisodeConditions format from the deleted PanelEnv era: `ambient_c`,
`wind_mps`, `alt`, Celsius. `ClosedLoopRuntime.calibrate()`, the actual
current API, needs grouped `weather`/`location` dicts with different
field names and Kelvin, not Celsius. Never actually triggered because
nothing wired `scenario_conditions_json()` into a real `calibrate_json()`
call yet, but would have silently produced wrong units or a KeyError the
moment something did.)

Each preset is ``{"description": str, "weather": {...}, "location": {...},
"time": {...}}`` -- ``weather``/``location`` go straight into
``calibrate(weather=..., location=...)``, ``time`` is a ready-made
``time_components`` argument for a first ``recommend()`` call afterward.

These are deliberately named after physically distinct regimes so a demo
audience can connect a scenario name to what they should expect to see
in the PINN-vs-RK4TRAN drift plot and the decision trace, and so they
loosely mirror the EDGE-side packet scenarios in
``workstation/scripts/simulate.py`` (``partly_cloudy``, ``fault_inject``, ...)
without being required to match one-for-one — the EDGE scenarios test
wire-level edge cases (nulls, faults), these test control-relevant
physical regimes.
"""

from __future__ import annotations

from typing import Any

SCENARIOS: dict[str, dict[str, Any]] = {
    "clear_sky_noon": {
        "description": "Clear midday sun, light breeze — steady-state baseline.",
        "weather": {"T_amb": 302.15, "wind_speed": 2.5, "wind_dir": 200.0,
                    "humidity": 0.35, "irradiance": 1020.0, "cloud_cover": 0.02,
                    "pressure": 101100.0},
        "location": {"lat": 36.17, "lon": -86.78, "elevation": 180.0},
        "time": {"minute": 0.0, "hour": 12.0, "day_of_year": 172, "month": 6, "year": 2026},
    },
    "cloud_ramp": {
        "description": "Fast-moving partial cloud deck — high irradiance rate-of-change.",
        "weather": {"T_amb": 300.15, "wind_speed": 6.0, "wind_dir": 240.0,
                    "humidity": 0.55, "irradiance": 640.0, "cloud_cover": 0.55,
                    "pressure": 100800.0},
        "location": {"lat": 36.17, "lon": -86.78, "elevation": 180.0},
        "time": {"minute": 0.0, "hour": 14.0, "day_of_year": 172, "month": 6, "year": 2026},
    },
    "high_wind_transient": {
        "description": "Gusty conditions that stress the mechanical rate limits and cooling.",
        "weather": {"T_amb": 291.15, "wind_speed": 16.0, "wind_dir": 270.0,
                    "humidity": 0.20, "irradiance": 780.0, "cloud_cover": 0.15,
                    "pressure": 84500.0},
        "location": {"lat": 40.0, "lon": -105.3, "elevation": 1600.0},
        "time": {"minute": 0.0, "hour": 15.0, "day_of_year": 100, "month": 4, "year": 2026},
    },
    "low_light_dawn": {
        "description": "Low sun angle shortly after sunrise — tests glancing incidence.",
        "weather": {"T_amb": 292.15, "wind_speed": 1.0, "wind_dir": 90.0,
                    "humidity": 0.70, "irradiance": 210.0, "cloud_cover": 0.10,
                    "pressure": 101300.0},
        "location": {"lat": 36.17, "lon": -86.78, "elevation": 180.0},
        "time": {"minute": 0.0, "hour": 6.5, "day_of_year": 172, "month": 6, "year": 2026},
    },
    "hot_desert_noon": {
        "description": "High ambient temperature stresses the thermal derating model.",
        "weather": {"T_amb": 317.15, "wind_speed": 3.0, "wind_dir": 180.0,
                    "humidity": 0.08, "irradiance": 1080.0, "cloud_cover": 0.0,
                    "pressure": 100500.0},
        "location": {"lat": 33.4, "lon": -112.1, "elevation": 330.0},
        "time": {"minute": 0.0, "hour": 13.0, "day_of_year": 200, "month": 7, "year": 2026},
    },
    "night_standby": {
        "description": "No usable irradiance — panel should settle to a park pose.",
        "weather": {"T_amb": 289.15, "wind_speed": 1.5, "wind_dir": 150.0,
                    "humidity": 0.60, "irradiance": 0.0, "cloud_cover": 0.30,
                    "pressure": 101400.0},
        "location": {"lat": 36.17, "lon": -86.78, "elevation": 180.0},
        "time": {"minute": 0.0, "hour": 23.0, "day_of_year": 172, "month": 6, "year": 2026},
    },
}


def list_scenarios() -> list[str]:
    """Return scenario names in a stable, demo-friendly order."""
    return list(SCENARIOS.keys())


def get_scenario(name: str) -> dict[str, Any]:
    """Return a copy of the named preset (description + weather + location + time)."""
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario {name!r}; available: {list_scenarios()}")
    return dict(SCENARIOS[name])
