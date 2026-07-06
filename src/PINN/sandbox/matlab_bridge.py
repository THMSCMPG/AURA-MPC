"""MATLAB-facing bridge for the closed-loop sandbox runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .runtime import ClosedLoopRuntime


def _loads_maybe(value: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("Expected a JSON object")
    return parsed


class MatlabSimulationBridge:
    """Thin JSON-string API so MATLAB can drive the Python runtime."""

    def __init__(
        self,
        config_path: str,
        pinn_checkpoint: str | None = None,
        policy_checkpoint: str | None = None,
        output_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        cfg_path = Path(config_path).resolve()
        with cfg_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        base_dir = cfg_path.parent.parent
        if pinn_checkpoint is None:
            pinn_checkpoint = str((base_dir / "outputs" / "pretrain" / "checkpoints" / "best_model.pt").resolve())
        if output_dir is None:
            output_dir = str((base_dir / "outputs" / "simulation").resolve())

        fortran_cfg = config.get("fortran", {})
        if "binary_path" in fortran_cfg:
            binary_path = Path(fortran_cfg["binary_path"])
            if not binary_path.is_absolute():
                fortran_cfg["binary_path"] = str((cfg_path.parent / binary_path).resolve())

        self.runtime = ClosedLoopRuntime(
            config=config,
            pinn_checkpoint=pinn_checkpoint,
            policy_checkpoint=policy_checkpoint,
            output_dir=output_dir,
            device=device,
        )

    def reset(
        self,
        initial_conditions_json: str = "",
        initial_pose_json: str = "",
        seed: int | None = None,
    ) -> str:
        record = self.runtime.reset(
            initial_conditions=_loads_maybe(initial_conditions_json),
            initial_pose=_loads_maybe(initial_pose_json),
            seed=None if seed is None else int(seed),
        )
        return json.dumps(record)

    def step(
        self,
        action_override_json: str = "",
        policy_mode: str = "mean",
        learning_enabled: bool = True,
        validate_with_rk4: bool | None = None,
    ) -> str:
        action_override = None
        if str(action_override_json).strip():
            parsed = json.loads(str(action_override_json))
            if not isinstance(parsed, list):
                raise TypeError("Action override must be a JSON array")
            action_override = [float(v) for v in parsed]
        record = self.runtime.step(
            action_override=action_override,
            policy_mode=str(policy_mode),
            learning_enabled=bool(learning_enabled),
            validate_with_rk4=validate_with_rk4,
        )
        return json.dumps(record)

    def latest_json(self) -> str:
        return json.dumps(self.runtime.latest())

    def history_json(self) -> str:
        return json.dumps(self.runtime.history())
