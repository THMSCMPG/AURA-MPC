"""MATLAB-facing bridge for the closed-loop sandbox runtime."""

from __future__ import annotations

import json
import subprocess
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

def _resolve_fortran_binary(cfg_path: Path, configured_path: str) -> Path:
    binary_path = Path(configured_path)
    if binary_path.is_absolute():
        return binary_path
    pinn_root = cfg_path.parent.parent
    src_root = pinn_root.parent
    candidates = [
        (cfg_path.parent / binary_path).resolve(),
        (pinn_root / binary_path).resolve(),
        (src_root / binary_path).resolve(),
        (src_root / "RK4TRAN" / binary_path.name).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def _ensure_fortran_binary(binary_path: Path) -> Path:
    if binary_path.exists():
        return binary_path
    rk4tran_dir = binary_path.parent
    make_script = rk4tran_dir / "make.sh"
    if make_script.exists():
        proc = subprocess.run(
            [str(make_script)],
            cwd=str(rk4tran_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Failed to build RK4TRAN evaluator.\n"
                f"Expected binary: {binary_path}\n"
                f"Build output:\n{proc.stdout}\n{proc.stderr}"
            )
    if not binary_path.exists():
        raise FileNotFoundError(
            "RK4TRAN evaluator binary was not found.\n"
            f"Expected path: {binary_path}\n"
            "If this is a fresh checkout, build src/RK4TRAN first."
        )
    return binary_path

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
            resolved_binary = _resolve_fortran_binary(cfg_path, str(fortran_cfg["binary_path"]))
            fortran_cfg["binary_path"] = str(_ensure_fortran_binary(resolved_binary))

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
