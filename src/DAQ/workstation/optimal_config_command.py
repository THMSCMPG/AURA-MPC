"""workstation/optimal_config_command.py – Output schema for the inference layer.

Defines the JSON schema that the workstation returns to the Pi 3B+ gateway
after every optimisation cycle.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

OPTIMAL_CONFIG_COMMAND_SCHEMA: dict = {
    "timestamp_iso": str,
    "tilt_opt": float,          # degrees  [0, 90]
    "azimuth_opt": float,       # degrees  [-180, 180]
    "height_opt": float,        # metres   [0.5, 5.0]
    "P_mp_pred": float,         # predicted max power (W)
    "T_panel_pred": float,      # predicted panel temperature (°C)
    "solver_used": str,         # which Fortran tier was routed to
    "n_pso_iterations": int,
    "optimiser": str,           # "pso" | "bo"
    "inference_time_ms": float,
    "workstation_version": str,
}

# Ordered list of field names for easy iteration.
OPTIMAL_CONFIG_COMMAND_FIELDS: tuple[str, ...] = tuple(
    OPTIMAL_CONFIG_COMMAND_SCHEMA.keys()
)

WORKSTATION_VERSION: str = "1.0.0"
