"""MPC candidate-evaluation runtime for live/replay panel-orientation decisions.

REWORKED from the original RL-based ClosedLoopRuntime (see checklist Section
1.12 for the full decision history). The old version wrapped PanelEnv (a
Gym-style environment with reward shaping) and PolicyNetwork (a
policy-gradient-trained action distribution), auto-updating policy weights
after each episode. Per the sim-to-real scope decision this session: train
purely on RK4TRAIN's synthetic data, no live weight updates, keep the
candidate-evaluation logic itself (that's the actual MPC, not RL scaffolding)
and add a light predicted-vs-actual outcome log instead of a reward signal.

What changed concretely:
  - No PolicyNetwork. MPC doesn't need a LEARNED proposal distribution --
    only a way to generate candidates and a trained forward model (PINN) to
    evaluate them. Candidates are sampled uniformly within the symmetry-
    reduced orientation domain (pitch, yaw in [0,90], roll fixed at 0 --
    same reduction used for the 12x12 lattice grid, see D5).
  - No PanelEnv, no reward, no discounted returns, no gradient step. Decision
    = argmax predicted cooling over N candidates (D6: pure argmax, no
    confidence gating -- but confidence is fully logged for every candidate,
    not just the winner, per D6's "self-graded confidence" framing for
    later paper analysis).
  - Selection uses PINNSurrogate's fast 15-min transient prediction
    (T_after_15min, D10) directly -- NOT the slower live Fortran evaluator
    per-candidate. The Fortran evaluator (via RK4TRANValidator, wrapped in
    SandboxPINNAgent) still runs ONCE per decision for steady-state
    cross-validation/bias-correction, same as before.
  - No auto-actuation. recommend() returns a recommendation; you move the
    panel; record_outcome() logs what actually happened afterward. This
    matches the calibration workflow: manual position confirm, manual
    weather/irradiance entry (camera dropped, no live cloud-cover sensing --
    see checklist Section 3 notes).
  - matlab_bridge.py's role (kept, per explicit instruction) is replay-
    after-the-fact validation of logged sessions, not live orchestration --
    this class is what a live session (or a replay) drives.

HONEST LIMITATION, not hidden: candidate ranking uses PINN's own
T_after_15min prediction with no independent cross-check, because no live
Fortran transient evaluator exists yet (evaluate_state.f90 only computes
steady state). See SandboxPINNAgent.predict()'s docstring for the same note.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .integration import SandboxPINNAgent

# Symmetry-reduced orientation domain -- same reduction used for the
# RK4TRAIN lattice's 12x12 (pitch, yaw) grid (D5). Roll is not swept (fixed
# at 0) -- it only ever attenuates capture in this model, so 0 is also the
# electrically-sensible fixed choice, consistent with the lattice generator.
PITCH_BOUNDS = (0.0, 90.0)
YAW_BOUNDS = (0.0, 90.0)
FIXED_ROLL = 0.0


@dataclass
class CandidateResult:
    """One candidate orientation's PINN evaluation."""

    pitch: float
    roll: float
    yaw: float
    T_after_15min: float
    T_after_15min_sigma: float
    eta_after_15min: float
    eta_after_15min_sigma: float
    predicted_cooling: float  # T_panel_current - T_after_15min; larger = more cooling = better

    def to_dict(self) -> dict[str, float]:
        return {
            "pitch": self.pitch,
            "roll": self.roll,
            "yaw": self.yaw,
            "T_after_15min": self.T_after_15min,
            "T_after_15min_sigma": self.T_after_15min_sigma,
            "eta_after_15min": self.eta_after_15min,
            "eta_after_15min_sigma": self.eta_after_15min_sigma,
            "predicted_cooling": self.predicted_cooling,
        }


@dataclass
class SessionConditions:
    """Session-constant conditions, set at calibration -- weather/irradiance
    are now MANUAL entries (camera dropped, no live cloud-cover sensing;
    the two-stage PSO irradiance estimator was also cut this session, so
    irradiance is a manual set-value too, consistent with the weather-entry
    approach). Location is fixed once at calibration. T_amb/wind/pressure
    may still come from live DAQ sensors if wired up -- inject_conditions()
    lets those update mid-session without a full recalibration."""

    weather: dict[str, float]
    location: dict[str, float]
    panel_state_fixed: dict[str, float] = field(default_factory=lambda: {"pv_height": 1.5})


class ClosedLoopRuntime:
    """MPC runtime: propose candidate orientations, evaluate with the PINN's
    fast transient lookahead, recommend the best by argmax predicted
    cooling, log everything (including confidence) for later self-graded
    analysis. No auto-actuation, no weight updates.
    """

    def __init__(
        self,
        *,
        pinn_checkpoint: Path | str,
        output_dir: Path | str,
        rk4_binary: Optional[Path | str] = None,
        normalizer_path: Optional[Path | str] = None,
        device: str = "cpu",
        correction_alpha: float = 0.25,
        n_candidates: int = 12,
        rng_seed: Optional[int] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "closed_loop_trace.jsonl"
        self.trace_path.write_text("", encoding="utf-8")

        pinn_checkpoint = Path(pinn_checkpoint)
        normalizer_path = Path(normalizer_path) if normalizer_path else (pinn_checkpoint.parent / "normalizer.json")
        self.pinn_agent = SandboxPINNAgent(
            pinn_checkpoint=pinn_checkpoint,
            rk4_binary=rk4_binary,
            normalizer_path=normalizer_path if Path(normalizer_path).exists() else None,
            device=device,
            correction_alpha=correction_alpha,
        )

        self.n_candidates = int(n_candidates)
        self._rng = np.random.default_rng(rng_seed)
        self.conditions: Optional[SessionConditions] = None
        self._history: list[dict[str, Any]] = []
        self._decision_counter = 0

    def _append_trace(self, record: dict[str, Any]) -> None:
        self._history.append(record)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(
        self,
        *,
        weather: dict[str, float],
        location: dict[str, float],
        target_position: dict[str, float],
        confirmed_position: dict[str, float],
        confirmed_readback: dict[str, float],
    ) -> dict[str, Any]:
        """Session-start calibration handshake: report a target position,
        operator physically sets it, operator confirms independent-variable
        readback matches. This method records the handshake -- it does NOT
        drive any hardware (no EDGE wiring here); the actual "tell the
        operator, wait for confirmation" interaction is a CLI/UI concern one
        layer up.

        Args:
            weather: manually-entered session-constant weather (see
                SessionConditions -- irradiance/cloud_cover are set-value
                entries now, not sensed or estimated)
            location: fixed session location
            target_position: the position the operator was asked to set
            confirmed_position: what the operator reports they actually set
                (should match target_position; logged either way -- a
                mismatch here is itself useful information, not an error)
            confirmed_readback: independent variables read back after
                positioning, for the operator to visually confirm look right
        """
        self.conditions = SessionConditions(weather=dict(weather), location=dict(location))
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "calibrate",
            "weather": weather,
            "location": location,
            "target_position": target_position,
            "confirmed_position": confirmed_position,
            "confirmed_readback": confirmed_readback,
            "position_matches_target": target_position == confirmed_position,
        }
        self._append_trace(record)
        return record

    def inject_conditions(self, weather: Optional[dict[str, float]] = None, location: Optional[dict[str, float]] = None) -> None:
        """Update session conditions mid-session (e.g. a live DAQ T_amb/wind
        reading, or a manual weather re-entry) without a full recalibration.
        """
        if self.conditions is None:
            raise RuntimeError("Cannot inject conditions before calibrate() has been called")
        if weather is not None:
            self.conditions.weather.update(weather)
        if location is not None:
            self.conditions.location.update(location)

    # ------------------------------------------------------------------
    # Candidate generation + evaluation + recommendation
    # ------------------------------------------------------------------
    def propose_candidates(self, n: Optional[int] = None) -> list[tuple[float, float, float]]:
        """Uniform random (pitch, roll=0, yaw) samples within the symmetry-
        reduced domain. No learned proposal distribution -- MPC only needs
        candidates to evaluate, not a trained policy to draw them from
        (see module docstring)."""
        n = n if n is not None else self.n_candidates
        pitches = self._rng.uniform(PITCH_BOUNDS[0], PITCH_BOUNDS[1], size=n)
        yaws = self._rng.uniform(YAW_BOUNDS[0], YAW_BOUNDS[1], size=n)
        return [(float(p), FIXED_ROLL, float(y)) for p, y in zip(pitches, yaws)]

    def evaluate_candidates(
        self,
        candidates: list[tuple[float, float, float]],
        T_panel_current: float,
        time_components: dict[str, float],
    ) -> list[CandidateResult]:
        """Runs the PINN's fast transient lookahead for every candidate.
        Does NOT call the (slower, steady-state-only) live Fortran
        evaluator per candidate -- that's the whole point of D10's
        transient prediction heads."""
        if self.conditions is None:
            raise RuntimeError("Cannot evaluate candidates before calibrate() has been called")

        results: list[CandidateResult] = []
        for pitch, roll, yaw in candidates:
            panel_state = dict(self.conditions.panel_state_fixed)
            panel_state.update({"pitch": pitch, "roll": roll, "yaw": yaw})

            out = self.pinn_agent.predict(
                weather=self.conditions.weather,
                panel_state=panel_state,
                location=self.conditions.location,
                time_components=time_components,
                T_panel_initial=T_panel_current,
                include_rk4=False,  # per-candidate: PINN only, see module docstring's honest limitation note
            )
            pinn_out = out["pinn"]
            cooling = T_panel_current - pinn_out["T_after_15min"]
            results.append(CandidateResult(
                pitch=pitch, roll=roll, yaw=yaw,
                T_after_15min=pinn_out["T_after_15min"],
                T_after_15min_sigma=pinn_out["T_after_15min_sigma"],
                eta_after_15min=pinn_out["eta_after_15min"],
                eta_after_15min_sigma=pinn_out["eta_after_15min_sigma"],
                predicted_cooling=cooling,
            ))
        return results

    def recommend(
        self,
        *,
        T_panel_current: float,
        time_components: dict[str, float],
        n_candidates: Optional[int] = None,
    ) -> dict[str, Any]:
        """Full decision cycle: propose -> evaluate -> argmax -> cross-
        validate steady-state against RK4TRAN once -> log everything -> return
        the recommendation. Does NOT actuate anything -- the caller is
        responsible for having the operator move the panel and later calling
        record_outcome().
        """
        if self.conditions is None:
            raise RuntimeError("Cannot recommend before calibrate() has been called")

        candidates = self.propose_candidates(n_candidates)
        results = self.evaluate_candidates(candidates, T_panel_current, time_components)

        # Pure argmax by predicted cooling (D6: no confidence gating) --
        # but every candidate's confidence is logged, not just the winner's.
        best_idx = max(range(len(results)), key=lambda i: results[i].predicted_cooling)
        best = results[best_idx]

        # One steady-state RK4TRAN cross-check at the CHOSEN candidate's
        # orientation, for the running bias-correction/drift-metrics
        # machinery already in SandboxPINNAgent (unchanged from before).
        chosen_panel_state = dict(self.conditions.panel_state_fixed)
        chosen_panel_state.update({"pitch": best.pitch, "roll": best.roll, "yaw": best.yaw})
        rk4_check = self.pinn_agent.predict(
            weather=self.conditions.weather,
            panel_state=chosen_panel_state,
            location=self.conditions.location,
            time_components=time_components,
            T_panel_initial=T_panel_current,
            include_rk4=self.pinn_agent.rk4 is not None,
        )

        self._decision_counter += 1
        decision_id = f"decision_{self._decision_counter:05d}"
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "recommend",
            "decision_id": decision_id,
            "T_panel_current": T_panel_current,
            "time_components": time_components,
            "weather": dict(self.conditions.weather),
            "location": dict(self.conditions.location),
            "candidates": [r.to_dict() for r in results],
            "chosen_index": best_idx,
            "chosen": best.to_dict(),
            "rk4_steady_state_check": rk4_check["rk4"],
            "discrepancy": rk4_check["discrepancy"],
            "outcome": None,  # filled in later by record_outcome()
        }
        self._append_trace(record)
        return record

    # ------------------------------------------------------------------
    # Light predicted-vs-actual logging (self-graded confidence record)
    # ------------------------------------------------------------------
    def record_outcome(
        self,
        decision_id: str,
        *,
        efficiency_before: float,
        efficiency_after: float,
    ) -> dict[str, Any]:
        """Call this after the operator has moved the panel and new sensor
        data has come in. Compares real pre/post electrical efficiency --
        deliberately NOT connected to the model (per the original design:
        "a separate code that isn't connected to the model to assess its
        decision"). eta up = good, eta down = bad. This IS the light
        predicted-vs-actual logging layer -- a self-graded confidence
        record for later analysis (does the model's stated confidence
        actually track its real-world accuracy?), not a training signal.
        No weights are touched here.
        """
        good = efficiency_after > efficiency_before
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "outcome",
            "decision_id": decision_id,
            "efficiency_before": efficiency_before,
            "efficiency_after": efficiency_after,
            "efficiency_delta": efficiency_after - efficiency_before,
            "good": good,
        }
        self._append_trace(record)
        return record

    def latest(self) -> dict[str, Any]:
        return dict(self._history[-1]) if self._history else {}

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
