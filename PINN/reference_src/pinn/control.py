"""Panel control optimizer for PINN-AURA-MFP (Batch D — Day 8).

Public API:

* :class:`OrchestrationCommand` — immutable JSON-serialisable command
  emitted by the orchestrator's transmit path (design doc §6.5).
* :class:`PanelControlOptimizer` — deterministic, stateless optimizer
  converting PINN predictions into physically realisable pitch / yaw /
  roll / z setpoints (design doc §6.4).
* :func:`apply_watchdog` — binary fallback trigger (design doc §5.1).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from ..config import ControlConfig, ProductContract
from ..utils.logging import get_logger
from .data import SensorPacket

_LOGGER = get_logger(__name__)

# Default initial pose used when no ``current_pose`` is supplied.
_DEFAULT_INITIAL_POSE: dict[str, float] = {
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "z": 1.0,
}

# Keys of the four-DoF pose / command vector, in canonical order.
_POSE_KEYS: tuple[str, ...] = ("pitch", "yaw", "roll", "z")


def _clip(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to ``[lo, hi]`` using plain Python floats."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _clip_with_log(
    value: float, lo: float, hi: float, name: str
) -> float:
    """Clamp and log at DEBUG when the value was actually constrained."""
    clipped = _clip(value, lo, hi)
    if clipped != value:
        _LOGGER.debug(
            "control clamp applied",
            extra={"field": name, "raw": float(value), "clipped": float(clipped),
                   "lo": float(lo), "hi": float(hi)},
        )
    return clipped


# --------------------------------------------------------------------- #
# 8.1  OrchestrationCommand                                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OrchestrationCommand:
    """Immutable transmit-path command (design doc §6.5).

    Attributes:
        sim_mode: Simulation routing mode. Currently only
            ``"route_labels"`` is supported.
        aura_flag: ``True`` when AURA-MFP is the solver of record
            (always ``True`` in this codebase).
        pitch: Pitch setpoint (deg).
        yaw: Yaw setpoint (deg).
        roll: Roll setpoint (deg).
        z: Panel-height setpoint (m).
        predicted_temp: PINN-predicted panel temperature (K).
        uncertainty: PINN predictive uncertainty in ``[0, 1]``.
        fallback_active: ``True`` if the watchdog has tripped.
        timestamp: Wall-clock timestamp at command generation.
    """

    sim_mode: str
    aura_flag: bool
    pitch: float
    yaw: float
    roll: float
    z: float
    predicted_temp: float
    uncertainty: float
    fallback_active: bool
    timestamp: datetime

    def to_json(self) -> str:
        """Serialise the command as a stable, sorted JSON string.

        ``timestamp`` is emitted in ISO-8601 format so the
        :meth:`from_json` / :meth:`to_json` round-trip is bit-exact for
        deterministic inputs.
        """
        payload: dict[str, Any] = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "OrchestrationCommand":
        """Deserialise a command previously produced by :meth:`to_json`.

        Args:
            s: JSON string.

        Returns:
            The reconstructed :class:`OrchestrationCommand`.

        Raises:
            ValueError: If ``s`` is missing a required field or has a
                malformed timestamp.
        """
        raw = json.loads(s)
        if not isinstance(raw, dict):
            raise ValueError("OrchestrationCommand JSON must decode to an object")
        try:
            ts = datetime.fromisoformat(str(raw["timestamp"]))
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid or missing 'timestamp': {exc}") from exc
        required = {
            "sim_mode",
            "aura_flag",
            "pitch",
            "yaw",
            "roll",
            "z",
            "predicted_temp",
            "uncertainty",
            "fallback_active",
        }
        missing = required - set(raw)
        if missing:
            raise ValueError(
                f"OrchestrationCommand JSON missing fields: {sorted(missing)}"
            )
        return cls(
            sim_mode=str(raw["sim_mode"]),
            aura_flag=bool(raw["aura_flag"]),
            pitch=float(raw["pitch"]),
            yaw=float(raw["yaw"]),
            roll=float(raw["roll"]),
            z=float(raw["z"]),
            predicted_temp=float(raw["predicted_temp"]),
            uncertainty=float(raw["uncertainty"]),
            fallback_active=bool(raw["fallback_active"]),
            timestamp=ts,
        )


# --------------------------------------------------------------------- #
# 8.2  PanelControlOptimizer                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PanelControlOptimizer:
    """Deterministic panel-pose optimizer.

    The optimizer has **no mutable state**: successive calls with
    identical arguments produce identical outputs. All tuning is pulled
    from the :class:`ControlConfig` supplied at construction.

    The target formulas follow design doc §6.4 exactly:

    * ``target_pitch = clip(0.4·WS − 0.02·T_pred, −20, 20)``
    * ``target_yaw   = clip(cloud_azimuth_bias · 45, −120, 120)`` where
      ``cloud_azimuth_bias = sign(sin(2π·t_s/86400)) · CC`` — a
      placeholder heuristic documented in the design doc.
    * ``target_roll  = clip(0.15·WS, −10, 10)``
    * ``target_z     = clip(1.0 + 0.8·WS/15 + 0.4·CC, 0, 3)``

    Targets are then slew-rate-limited against ``current_pose`` using the
    per-axis rates in :class:`ControlConfig`, and finally clamped to the
    absolute bounds. Every clamp is logged at DEBUG.
    """

    control_cfg: ControlConfig
    dt: float = 1.0

    # ---- target formulas (design doc §6.4) ---------------------------

    @staticmethod
    def _cloud_azimuth_bias(t_s: float, CC: float) -> float:
        """Placeholder heuristic: ``sign(sin(2π·t_s/86400)) · CC``.

        Documented in the design doc §6.4 as a stopgap until a real
        sky-image-driven bias is wired in (Batch F).
        """
        s = math.sin(2.0 * math.pi * float(t_s) / 86400.0)
        # math.copysign(1, 0.0) == 1.0; we want a true zero bias when s == 0
        if s == 0.0:
            sign = 0.0
        elif s > 0.0:
            sign = 1.0
        else:
            sign = -1.0
        return sign * float(CC)

    def _targets(
        self, predicted_temp: float, sensor_state: SensorPacket
    ) -> dict[str, float]:
        """Compute raw pose targets from the PINN prediction and sensors."""
        WS = float(sensor_state.WS)
        CC = float(sensor_state.CC)
        t_s = float(sensor_state.t_s)
        T_pred = float(predicted_temp)

        target_pitch = _clip_with_log(
            0.4 * WS - 0.02 * T_pred, -20.0, 20.0, "target_pitch"
        )
        bias = self._cloud_azimuth_bias(t_s, CC)
        target_yaw = _clip_with_log(bias * 45.0, -120.0, 120.0, "target_yaw")
        target_roll = _clip_with_log(0.15 * WS, -10.0, 10.0, "target_roll")
        target_z = _clip_with_log(
            1.0 + 0.8 * (WS / 15.0) + 0.4 * CC, 0.0, 3.0, "target_z"
        )
        return {
            "pitch": target_pitch,
            "yaw": target_yaw,
            "roll": target_roll,
            "z": target_z,
        }

    def _resolve_current_pose(
        self, current_pose: dict[str, float] | None
    ) -> dict[str, float]:
        """Return a validated pose dict; default to the initial pose."""
        if current_pose is None:
            return dict(_DEFAULT_INITIAL_POSE)
        resolved: dict[str, float] = {}
        for key in _POSE_KEYS:
            if key not in current_pose:
                raise ValueError(f"current_pose missing key '{key}'")
            resolved[key] = float(current_pose[key])
        return resolved

    def optimize(
        self,
        predicted_temp: float,
        sensor_state: SensorPacket,
        current_pose: dict[str, float] | None,
        pose_override: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Produce a new pose setpoint from prediction + sensors.

        Args:
            predicted_temp: PINN panel-temperature prediction (same units
                the PINN emits, typically K or °C — the target formula
                is agnostic to the offset since it is linear).
            sensor_state: Latest validated :class:`SensorPacket`.
            current_pose: Current pose dict with keys
                ``{"pitch", "yaw", "roll", "z"}``. ``None`` means "assume
                the initial pose ``(0, 0, 0, 1.0)``".

        Returns:
            ``{"pitch", "yaw", "roll", "z"}`` — the new pose setpoint,
            guaranteed to satisfy both slew-rate and absolute bounds.
        """
        cfg = self.control_cfg
        dt = float(self.dt)
        current = self._resolve_current_pose(current_pose)
        # When pose_override is provided (PINN pose head output), use it as
        # the target directly; otherwise fall back to the heuristic formulas.
        if pose_override is not None:
            targets = {
                "pitch": float(pose_override.get("pitch", 0.0)),
                "yaw":   float(pose_override.get("yaw",   0.0)),
                "roll":  float(pose_override.get("roll",  0.0)),
                "z":     float(pose_override.get("z",     1.0)),
            }
        else:
            targets = self._targets(predicted_temp, sensor_state)

        # Per-axis slew-rate limits.
        rate_limits: dict[str, float] = {
            "pitch": cfg.max_pitch_rate,
            "yaw": cfg.max_yaw_rate,
            "roll": cfg.max_roll_rate,
            "z": cfg.max_z_rate,
        }
        # Absolute bounds as (lo, hi).
        abs_bounds: dict[str, tuple[float, float]] = {
            "pitch": (-cfg.pitch_bound, cfg.pitch_bound),
            "yaw": (-cfg.yaw_bound, cfg.yaw_bound),
            "roll": (-cfg.roll_bound, cfg.roll_bound),
            "z": (cfg.z_min, cfg.z_max),
        }

        new_pose: dict[str, float] = {}
        for key in _POSE_KEYS:
            delta = targets[key] - current[key]
            max_step = rate_limits[key] * dt
            limited_delta = _clip_with_log(
                delta, -max_step, max_step, f"slew_{key}"
            )
            candidate = current[key] + limited_delta
            lo, hi = abs_bounds[key]
            new_pose[key] = _clip_with_log(candidate, lo, hi, f"bound_{key}")

        return new_pose


# --------------------------------------------------------------------- #
# 8.3  apply_watchdog                                                   #
# --------------------------------------------------------------------- #


def apply_watchdog(
    uncertainty: float,
    fault_count: int,
    contract: ProductContract,
) -> bool:
    """Return ``True`` when the fallback path should engage.

    Implements design doc §5.1:

    * ``uncertainty > contract.uncertainty_watchdog`` **OR**
    * ``fault_count >= contract.max_consecutive_faults``.

    Args:
        uncertainty: Predictive uncertainty in ``[0, 1]``.
        fault_count: Number of consecutive ingest faults observed.
        contract: Operational :class:`ProductContract`.

    Returns:
        ``True`` if either trigger fires, else ``False``.
    """
    return (
        float(uncertainty) > float(contract.uncertainty_watchdog)
        or int(fault_count) >= int(contract.max_consecutive_faults)
    )
