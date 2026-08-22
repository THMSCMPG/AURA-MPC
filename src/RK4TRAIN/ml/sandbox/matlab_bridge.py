"""MATLAB-facing bridge for the MPC runtime -- primarily session replay now,
see MatlabSimulationBridge's docstring."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from . import scenarios as _scenarios
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
    ml_root = cfg_path.parent.parent           # .../RK4TRAIN/ml
    rk4train_root = ml_root.parent              # .../RK4TRAIN (Fortran generator + ml/ live here)
    candidates = [
        (cfg_path.parent / binary_path).resolve(),
        (ml_root / binary_path).resolve(),
        (rk4train_root / binary_path).resolve(),
        (rk4train_root / binary_path.name).resolve(),
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
                "Failed to build RK4TRAIN evaluator.\n"
                f"Expected binary: {binary_path}\n"
                f"Build output:\n{proc.stdout}\n{proc.stderr}"
            )
    if not binary_path.exists():
        raise FileNotFoundError(
            "RK4TRAIN evaluator binary was not found.\n"
            f"Expected path: {binary_path}\n"
            "If this is a fresh checkout, build src/RK4TRAIN first."
        )
    return binary_path

class MatlabSimulationBridge:
    """Thin JSON-string API so MATLAB can drive/replay the Python runtime.

    REWORKED (see checklist Section 1.12/3.7 for the decision history):
    this bridge's role shifted from "live RL sandbox driver" to "replay
    finished sessions for offline validation" -- the live decision loop is
    now driven by decision_server.py (EDGE-facing), not MATLAB. This class
    still exposes live calibrate/recommend/record_outcome methods (fixed to
    match ClosedLoopRuntime's new MPC API, they were broken against it after
    the RL removal) in case MATLAB-side ad hoc driving is still useful, but
    the NEW, primary capability is replay_session_json() -- load a finished
    session's logged trace (closed_loop_trace.jsonl) for MATLAB-side
    visualization/validation after the fact.
    """

    def __init__(
        self,
        config_path: str,
        pinn_checkpoint: str | None = None,
        output_dir: str | None = None,
        device: str = "cpu",
        n_candidates: int = 12,
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
        rk4_binary = None
        if "binary_path" in fortran_cfg:
            resolved_binary = _resolve_fortran_binary(cfg_path, str(fortran_cfg["binary_path"]))
            rk4_binary = _ensure_fortran_binary(resolved_binary)

        self.runtime = ClosedLoopRuntime(
            pinn_checkpoint=pinn_checkpoint,
            output_dir=output_dir,
            rk4_binary=rk4_binary,
            device=device,
            n_candidates=n_candidates,
        )

    # ── Live driving (fixed to match the new MPC API) ──────────────────
    def calibrate_json(
        self,
        weather_json: str,
        location_json: str,
        target_position_json: str,
        confirmed_position_json: str,
        confirmed_readback_json: str,
    ) -> str:
        record = self.runtime.calibrate(
            weather=_loads_maybe(weather_json) or {},
            location=_loads_maybe(location_json) or {},
            target_position=_loads_maybe(target_position_json) or {},
            confirmed_position=_loads_maybe(confirmed_position_json) or {},
            confirmed_readback=_loads_maybe(confirmed_readback_json) or {},
        )
        return json.dumps(record)

    def recommend_json(self, T_panel_current: float, time_components_json: str, n_candidates: int | None = None) -> str:
        record = self.runtime.recommend(
            T_panel_current=float(T_panel_current),
            time_components=_loads_maybe(time_components_json) or {},
            n_candidates=None if n_candidates is None else int(n_candidates),
        )
        return json.dumps(record)

    def record_outcome_json(self, decision_id: str, efficiency_before: float, efficiency_after: float) -> str:
        record = self.runtime.record_outcome(
            str(decision_id),
            efficiency_before=float(efficiency_before),
            efficiency_after=float(efficiency_after),
        )
        return json.dumps(record)

    def latest_json(self) -> str:
        return json.dumps(self.runtime.latest())

    def history_json(self) -> str:
        return json.dumps(self.runtime.history())

    # ── Replay (the primary role now -- see class docstring) ───────────
    def replay_session_json(self, trace_path: str) -> str:
        """Load a finished live session's logged trace
        (ClosedLoopRuntime's closed_loop_trace.jsonl, or decision_server's
        richer edge_decision_log.jsonl) for MATLAB-side offline
        visualization/validation. Does NOT re-run any inference -- this is
        a pure log reader, matching the "replay after live sessions"
        design intent (checklist Section 3.7).

        Returns a JSON array of every record in the trace file, in order.
        """
        path = Path(trace_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return json.dumps(records)

    # ── Scenario presets (for the MATLAB "Scenario preset" dropdown) ──────
    def list_scenarios_json(self) -> str:
        """Return ``{name: description}`` for every named preset."""
        return json.dumps(
            {name: _scenarios.SCENARIOS[name].get("description", "") for name in _scenarios.list_scenarios()}
        )

    def scenario_conditions_json(self, name: str) -> str:
        """Return one preset's condition fields, ready to fill MATLAB's edit boxes."""
        return json.dumps(_scenarios.get_scenario(str(name)))

    # ── Live conditions injection (EDGE feed, or MATLAB-side "what if") ───
    def inject_conditions_json(self, weather_json: str = "", location_json: str = "") -> None:
        """Overwrite the running session's conditions ahead of the next
        ``recommend`` call. Mirrors ``ClosedLoopRuntime.inject_conditions``
        for MATLAB callers that want to drive the loop from a live EDGE
        packet (translated via ``sandbox.edge_adapter.edge_packet_to_conditions``)
        or from an ad hoc "what if the wind picked up" edit, without a full
        recalibration.
        """
        self.runtime.inject_conditions(
            weather=_loads_maybe(weather_json),
            location=_loads_maybe(location_json),
        )
