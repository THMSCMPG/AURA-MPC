"""Real-time orchestration loop for PINN-AURA-MFP (Batch E — Day 9).

Implements :class:`RealTimeOrchestrator`, the runtime pipeline that
consumes :class:`SensorPacket` objects, runs the trained dual-head PINN,
routes to AURA-MFP solvers asynchronously, and emits
:class:`OrchestrationCommand` transmit-path records.

The design follows the completion plan §6.5:

1. Ingest via :class:`UnifiedDataBuffer` (no blocking I/O, no exceptions
   propagate out of the hot path).
2. Forward pass through :class:`DualHeadPINN.predict_with_uncertainty`.
3. Binary watchdog (:func:`apply_watchdog`) — on trip, emit a fallback
   command sourced from the last-known-good pose.
4. Dispatch the argmax-selected solver tier to a background thread pool
   (non-blocking) — results are stored in ``last_forecast`` keyed by
   tier, consumed by subsequent ``step()`` calls.
5. Deterministic :class:`PanelControlOptimizer` produces the next pose.
6. Build and return the :class:`OrchestrationCommand`.

The hot path (ingest → infer → control → emit) is required to complete
in ≤ 1 s on CPU. Solver dispatch happens in the background and never
blocks ``step()``.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess  # noqa: S404 — subprocess is required to talk to AURA-MFP binaries
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ..config import AppConfig
from ..utils.logging import get_logger
from .control import (
    OrchestrationCommand,
    PanelControlOptimizer,
    apply_watchdog,
)
from .data import SensorPacket, UnifiedDataBuffer
from .model import DualHeadPINN
from .physics import LearnedPhysicsParameters

_LOGGER = get_logger(__name__)

# Per-tier subprocess timeouts (seconds). Tiers match
# :attr:`ModelConfig.route_labels`.
_DEFAULT_TIER_TIMEOUTS: dict[str, float] = {
    "LOFI":  10.0,
    "SIMV1": 15.0,
    "SIMV2": 10.0,
    "SIMV3": 10.0,
    "SIMV4": 10.0,
}

# Number of recent step() wall times to retain for status reporting.
_STATUS_WINDOW: int = 100

# Pose keys / initial pose match :mod:`src.pinn.control`.
_POSE_KEYS: tuple[str, ...] = ("pitch", "yaw", "roll", "z")
_DEFAULT_INITIAL_POSE: dict[str, float] = {
    "pitch": 0.0,
    "yaw": 0.0,
    "roll": 0.0,
    "z": 1.0,
}


def build_fallback_command(
    last_command: OrchestrationCommand | None,
    reason: str,
    *,
    uncertainty: float | None = None,
    predicted_temp: float | None = None,
) -> OrchestrationCommand:
    """Build a graceful-degradation :class:`OrchestrationCommand`.

    Per design doc §5.1, the watchdog fallback emits ``sim_mode="SIMV4"``
    and carries forward the last-known-good pose. If no prior command is
    available, a zero pose ``(pitch=yaw=roll=0, z=0.0)`` is used.

    Args:
        last_command: The previous command emitted by the orchestrator,
            or ``None`` on cold start.
        reason: Human-readable reason for the fallback; logged at WARN.
        uncertainty: Current PINN uncertainty if available; defaults to
            ``1.0`` (maximum) when not provided.
        predicted_temp: Current PINN temperature projection if
            available; defaults to ``last_command.predicted_temp`` then
            ``0.0``.

    Returns:
        A fallback :class:`OrchestrationCommand` with
        ``fallback_active=True``.
    """
    _LOGGER.warning("fallback command emitted", extra={"reason": reason})
    if last_command is not None:
        pitch = last_command.pitch
        yaw = last_command.yaw
        roll = last_command.roll
        z = last_command.z
        fallback_temp = last_command.predicted_temp
    else:
        pitch = yaw = roll = 0.0
        z = 0.0
        fallback_temp = 0.0
    return OrchestrationCommand(
        sim_mode="SIMV4",
        aura_flag=True,
        pitch=float(pitch),
        yaw=float(yaw),
        roll=float(roll),
        z=float(z),
        predicted_temp=float(predicted_temp if predicted_temp is not None else fallback_temp),
        uncertainty=float(uncertainty if uncertainty is not None else 1.0),
        fallback_active=True,
        timestamp=datetime.now(),
    )


def load_checkpoint_model(
    model_path: Path,
    config: AppConfig,
    device: torch.device | str = "cpu",
) -> DualHeadPINN:
    """Materialise a :class:`DualHeadPINN` and load weights from ``model_path``.

    Checkpoints written by :mod:`scripts.train` or
    :class:`CheckpointManager` contain a ``model_state`` key mapping to a
    ``state_dict``.

    Args:
        model_path: Path to the checkpoint ``.pt`` file.
        config: The :class:`AppConfig` governing architecture sizes.
        device: Device to move the model to.

    Returns:
        A :class:`DualHeadPINN` in ``eval`` mode on ``device``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        KeyError: If the checkpoint is missing ``model_state``.
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    physics = LearnedPhysicsParameters.from_physics_config(config.physics)
    model = DualHeadPINN(config.model, physics)
    # weights_only=False: checkpoints include optimizer + RNG state the
    # trainer produced. Trusted local file (we wrote it).
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    if "model_state" not in ck:
        raise KeyError(
            f"checkpoint {p} missing 'model_state' key; got keys {sorted(ck)}"
        )
    model.load_state_dict(ck["model_state"])
    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------- #
# Solver-dispatch worker                                                #
# --------------------------------------------------------------------- #


class _SolverJob:
    """One solver-dispatch request submitted to the background pool.

    Plain attribute container; kept private. Not a dataclass so we can
    cheaply drop it on the queue.
    """

    __slots__ = ("tier", "payload", "enqueued_at")

    def __init__(self, tier: str, payload: dict[str, Any]) -> None:
        self.tier = tier
        self.payload = payload
        self.enqueued_at = time.perf_counter()


class SolverDispatcher:
    """Thread-pool that runs AURA-MFP solver subprocesses off the hot path.

    The orchestrator submits :class:`_SolverJob` records to
    :meth:`submit`; the worker threads pick them up, invoke the
    appropriate ``bin/simvN`` binary with a JSON payload, parse the
    response, and store it in the public ``last_forecast`` dict keyed by
    tier name.

    The dispatcher uses a sentinel ``None`` on the queue to ask workers
    to exit; :meth:`stop` is idempotent.
    """

    def __init__(
        self,
        aura_mfp_root: Path,
        *,
        pool_size: int = 2,
        tier_timeouts: dict[str, float] | None = None,
        subprocess_runner: Any | None = None,
    ) -> None:
        """Create (but do not start) a dispatcher bound to ``aura_mfp_root``.

        Args:
            aura_mfp_root: Root of an AURA-MFP checkout containing
                ``src/<tier>/bin/<tier>`` binaries.
            pool_size: Number of worker threads. Must be ≥ 1.
            tier_timeouts: Per-tier subprocess timeouts. Missing tiers
                fall back to :data:`_DEFAULT_TIER_TIMEOUTS`.
            subprocess_runner: Optional callable matching
                :func:`subprocess.run`'s signature. Defaults to
                :func:`subprocess.run`. Injectable for testing.
        """
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        self.aura_mfp_root = Path(aura_mfp_root)
        self.pool_size = int(pool_size)
        self._timeouts = dict(_DEFAULT_TIER_TIMEOUTS)
        if tier_timeouts:
            self._timeouts.update(tier_timeouts)
        self._runner = subprocess_runner if subprocess_runner is not None else subprocess.run
        self._queue: "queue.Queue[_SolverJob | None]" = queue.Queue(maxsize=256)
        self._threads: list[threading.Thread] = []
        self._started = False
        self._stopping = threading.Event()
        self._state_lock = threading.Lock()

        # Public state read by the orchestrator.
        self.last_forecast: dict[str, dict[str, Any]] = {}
        self.last_solver_tier: str | None = None
        self.last_solver_wall_s: float | None = None
        self.last_solver_ok: bool | None = None

    # -- lifecycle -------------------------------------------------- #

    def start(self) -> None:
        """Spin up the worker threads; idempotent."""
        with self._state_lock:
            if self._started:
                return
            self._stopping.clear()
            for i in range(self.pool_size):
                t = threading.Thread(
                    target=self._worker_loop,
                    name=f"pinn-solver-{i}",
                    daemon=True,
                )
                t.start()
                self._threads.append(t)
            self._started = True

    def stop(self, timeout: float = 5.0) -> None:
        """Signal all workers to exit and join them.

        Args:
            timeout: Per-thread join timeout (seconds).
        """
        with self._state_lock:
            if not self._started:
                return
            self._stopping.set()
            # One sentinel per worker so every thread wakes.
            for _ in self._threads:
                try:
                    self._queue.put_nowait(None)
                except queue.Full:  # pragma: no cover - queue is sized generously
                    pass
            threads = list(self._threads)
            self._threads.clear()
            self._started = False
        for t in threads:
            t.join(timeout=timeout)

    # -- submission ------------------------------------------------- #

    def submit(self, tier: str, payload: dict[str, Any]) -> bool:
        """Enqueue a solver-dispatch request.

        Args:
            tier: Tier name (e.g. ``"SIMV1"``). Routes with no
                corresponding binary (``"LOFI"``) are ignored.
            payload: JSON-serialisable dict passed to the binary on
                stdin.

        Returns:
            ``True`` if the job was enqueued; ``False`` if it was
            skipped (no binary, not started, or queue full).
        """
        if not self._started:
            return False
        if not _has_binary(self.aura_mfp_root, tier):
            return False
        try:
            self._queue.put_nowait(_SolverJob(tier, payload))
            return True
        except queue.Full:
            _LOGGER.warning("solver-dispatch queue full; dropping", extra={"tier": tier})
            return False

    # -- internals -------------------------------------------------- #

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            try:
                job = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                self._run_job(job)
            except Exception as exc:  # noqa: BLE001 — must not kill the worker
                _LOGGER.warning(
                    "solver-dispatch job crashed",
                    extra={"tier": job.tier, "error": repr(exc)},
                )
            finally:
                self._queue.task_done()

    def _run_job(self, job: _SolverJob) -> None:
        binary = _binary_path(self.aura_mfp_root, job.tier)
        timeout = self._timeouts.get(job.tier, 10.0)
        payload_json = json.dumps(job.payload, sort_keys=True)
        t0 = time.perf_counter()
        ok = False
        try:
            proc = self._runner(  # noqa: S603 — operator-supplied binary path
                [str(binary)],
                input=payload_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            wall_s = time.perf_counter() - t0
            if proc.returncode != 0:
                _LOGGER.warning(
                    "solver exited non-zero",
                    extra={
                        "tier": job.tier,
                        "rc": int(proc.returncode),
                        "stderr": (proc.stderr or "")[:256],
                    },
                )
                return
            try:
                result = json.loads(proc.stdout)
            except json.JSONDecodeError as exc:
                _LOGGER.warning(
                    "solver emitted invalid JSON",
                    extra={"tier": job.tier, "error": str(exc)},
                )
                return
            if not isinstance(result, dict):
                _LOGGER.warning(
                    "solver output is not a JSON object",
                    extra={"tier": job.tier},
                )
                return
            self.last_forecast[job.tier] = result
            ok = True
        except subprocess.TimeoutExpired:
            wall_s = time.perf_counter() - t0
            _LOGGER.warning(
                "solver timed out",
                extra={"tier": job.tier, "timeout_s": float(timeout)},
            )
        except FileNotFoundError:
            wall_s = time.perf_counter() - t0
            _LOGGER.warning(
                "solver binary not found",
                extra={"tier": job.tier, "binary": str(binary)},
            )
        finally:
            self.last_solver_tier = job.tier
            self.last_solver_wall_s = float(wall_s) if "wall_s" in locals() else None
            self.last_solver_ok = ok
            _LOGGER.info(
                "solver-dispatch complete",
                extra={
                    "tier": job.tier,
                    "wall_s": self.last_solver_wall_s,
                    "ok": ok,
                },
            )


def _binary_path(aura_mfp_root: Path, tier: str) -> Path:
    """Return the expected AURA-MFP ``bin/<tier>`` path for ``tier``.

    Two layouts are supported, in priority order:

    1. Standalone AURA-MFP repo (legacy):  ``<root>/src/<tier>/bin/<tier>``.
    2. AURA-MPC monorepo:                  ``<root>/<tier>/bin/<tier>``
       (this is the layout under ``modules/aura-mfp/`` in the AURA-MPC
       monorepo — there is no intermediate ``src/`` segment).

    The first layout is preferred for backwards compatibility; if the
    binary does not exist there but the monorepo location has it, the
    monorepo location is returned.
    """
    tier_name = tier.lower()
    legacy = Path(aura_mfp_root) / "src" / tier_name / "bin" / tier_name
    monorepo = Path(aura_mfp_root) / tier_name / "bin" / tier_name
    if legacy.exists():
        return legacy
    if monorepo.exists():
        return monorepo
    # Default to legacy for the "missing binary" error path so existing
    # error messages keep pointing at the documented location.
    return legacy


def _has_binary(aura_mfp_root: Path | None, tier: str) -> bool:
    """Return ``True`` if a runnable binary exists for ``tier``."""
    if aura_mfp_root is None:
        return False
    if tier.upper() == "LOFI":
        # LOFI is PINN-only; no external binary dispatch.
        return False
    b = _binary_path(aura_mfp_root, tier)
    return b.exists() and os.access(b, os.X_OK)


def _packet_to_json(packet: SensorPacket) -> dict[str, Any]:
    """Serialise a :class:`SensorPacket` to the AURA-MFP JSON schema."""
    payload: dict[str, Any] = {
        "t_s": float(packet.t_s),
        "G_poa": float(packet.G_poa),
        "T_amb": float(packet.T_amb),
        "WS": float(packet.WS),
        "CC": float(packet.CC),
        "lat": float(packet.lat),
        "lon": float(packet.lon),
        "timestamp": packet.timestamp.isoformat(),
    }
    # V1 does not consume images but the contract accepts an empty slot.
    payload["img_features"] = []
    if packet.pose is not None:
        payload["pose"] = {k: float(packet.pose[k]) for k in _POSE_KEYS if k in packet.pose}
    return payload


# --------------------------------------------------------------------- #
# RealTimeOrchestrator                                                  #
# --------------------------------------------------------------------- #


class RealTimeOrchestrator:
    """Ingest → infer → route → optimise → emit runtime pipeline.

    The orchestrator owns:

    * a :class:`UnifiedDataBuffer` that validates and rolls sensor
      packets,
    * a loaded :class:`DualHeadPINN` in ``eval`` mode,
    * a :class:`PanelControlOptimizer`,
    * a :class:`SolverDispatcher` (optional; only when an AURA-MFP
      checkout path is provided).

    ``step()`` is the hot path: ingest → infer → watchdog → queue
    solver → optimise → emit. It never blocks on I/O. Solver dispatch
    runs on background threads; results are picked up on the next
    ``step()`` that consults ``last_forecast``.
    """

    def __init__(
        self,
        model_path: Path,
        config: AppConfig,
        aura_mfp_root: Path | None = None,
        *,
        device: torch.device | str = "cpu",
        model: DualHeadPINN | None = None,
        subprocess_runner: Any | None = None,
    ) -> None:
        """Load the model and wire the runtime subsystems.

        Args:
            model_path: Checkpoint file written by
                :mod:`scripts.train` / :class:`CheckpointManager`.
            config: Full :class:`AppConfig` controlling architecture
                sizes, watchdog thresholds, and control bounds.
            aura_mfp_root: Root of an AURA-MFP checkout. If ``None``,
                solver dispatch is disabled and the orchestrator runs
                in PINN-only mode.
            device: Torch device for inference.
            model: Pre-loaded model; if provided, ``model_path`` is not
                consulted. Intended for tests.
            subprocess_runner: Optional :func:`subprocess.run`-like
                callable for testing.
        """
        self.config = config
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.aura_mfp_root = Path(aura_mfp_root) if aura_mfp_root is not None else None

        if model is not None:
            self.model = model
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model = load_checkpoint_model(model_path, config, device=self.device)

        self.buffer = UnifiedDataBuffer(config.contract)
        self.optimizer = PanelControlOptimizer(
            control_cfg=config.control,
            dt=float(config.contract.decision_cadence_s),
        )

        self._current_pose: dict[str, float] = dict(_DEFAULT_INITIAL_POSE)
        self._last_command: OrchestrationCommand | None = None

        # Status / metrics.
        self._step_walls: deque[float] = deque(maxlen=_STATUS_WINDOW)
        self._fallback_flags: deque[bool] = deque(maxlen=_STATUS_WINDOW)
        self._state_lock = threading.Lock()

        self.dispatcher: SolverDispatcher | None = None
        if self.aura_mfp_root is not None:
            self.dispatcher = SolverDispatcher(
                self.aura_mfp_root,
                pool_size=2,
                subprocess_runner=subprocess_runner,
            )

    # -- lifecycle ------------------------------------------------- #

    def start(self) -> None:
        """Start the background solver-dispatch pool (if configured)."""
        if self.dispatcher is not None:
            self.dispatcher.start()

    def stop(self) -> None:
        """Tear down the solver-dispatch pool (if running)."""
        if self.dispatcher is not None:
            self.dispatcher.stop()

    def __enter__(self) -> "RealTimeOrchestrator":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # -- hot path -------------------------------------------------- #

    def step(self, packet: SensorPacket) -> OrchestrationCommand:
        """Run one end-to-end decision and return the emitted command.

        Non-blocking. Required to complete in ≤ 1 s on CPU.

        Args:
            packet: Incoming sensor reading.

        Returns:
            The :class:`OrchestrationCommand` for this decision cycle,
            which is also stashed as the last-known-good command for
            fallback reuse.
        """
        t0 = time.perf_counter()
        try:
            return self._step_impl(packet)
        finally:
            self._step_walls.append(time.perf_counter() - t0)

    def _step_impl(self, packet: SensorPacket) -> OrchestrationCommand:
        contract = self.config.contract

        # 1. Ingest. UnifiedDataBuffer.ingest swallows validation and
        # staleness errors and bumps its fault counter; we still use the
        # raw packet for inference so the watchdog can decide based on a
        # fresh forward pass, but we trust the buffer's fault tally.
        self.buffer.ingest(packet)
        fault_count = self.buffer.fault_count()

        # 2. Forward pass. We use the current packet's tensors regardless
        # of whether the buffer accepted it; if ingest failed due to a
        # malformed field, validate() below will raise and we short-circuit
        # to fallback with a generous uncertainty.
        try:
            packet.validate()
            tensors = packet.to_tensor()
            numeric = tensors["numeric"].unsqueeze(0).to(self.device)
            image = tensors["image"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.model.predict_with_uncertainty(numeric, image)
            t_hat = float(out["T_hat"].squeeze().item())
            uncertainty = float(out["route_uncertainty"].squeeze().item())
            route_probs = out["route_probs"].squeeze(0)
            route_idx = int(torch.argmax(route_probs).item())
            tier_name = self.config.model.route_labels[route_idx]
            inference_ok = True
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning(
                "inference failed; using fallback",
                extra={"error": repr(exc)},
            )
            t_hat = (
                self._last_command.predicted_temp
                if self._last_command is not None
                else 0.0
            )
            uncertainty = 1.0
            tier_name = "SIMV4"
            inference_ok = False

        # 3. Watchdog.
        tripped = apply_watchdog(uncertainty, fault_count, contract)
        if tripped or not inference_ok:
            reason = (
                f"uncertainty={uncertainty:.3f}>{contract.uncertainty_watchdog} "
                if uncertainty > contract.uncertainty_watchdog
                else f"fault_count={fault_count}>={contract.max_consecutive_faults} "
                if fault_count >= contract.max_consecutive_faults
                else "inference_failed"
            )
            cmd = build_fallback_command(
                self._last_command,
                reason,
                uncertainty=uncertainty,
                predicted_temp=t_hat,
            )
            with self._state_lock:
                self._last_command = cmd
                self._fallback_flags.append(True)
            return cmd

        # 4. Solver dispatch (non-blocking). PINN-only ("LOFI") is served
        # from the model's own T_hat; other tiers may be dispatched if a
        # local AURA-MFP checkout is available.
        if self.dispatcher is not None:
            self.dispatcher.submit(tier_name, _packet_to_json(packet))

        # 5. Consume most-recent forecast for the chosen tier if one has
        # arrived from a previous cycle; else use PINN's temperature.
        predicted_temp = t_hat
        if self.dispatcher is not None:
            forecast = self.dispatcher.last_forecast.get(tier_name)
            if forecast is not None and "T_panel" in forecast:
                try:
                    predicted_temp = float(forecast["T_panel"])
                except (TypeError, ValueError):
                    predicted_temp = t_hat

        # 6. Control optimisation.
        # Primary path: use the PINN's trained pose head output.
        # The pose head already encodes the physics (temperature-minimising,
        # irradiance-maximising) so we bypass the heuristic optimizer.
        # Slew-rate limiting and absolute bounds are still applied by
        # the PanelControlOptimizer's _resolve + slew logic.
        try:
            # out from step 2 above still has pose; re-run if inference_ok
            if inference_ok:
                tensors2 = packet.to_tensor()
                n2 = tensors2["numeric"].unsqueeze(0).to(self.device)
                i2 = tensors2["image"].unsqueeze(0).to(self.device)
                with torch.no_grad():
                    pinn_out = self.model.predict_with_uncertainty(n2, i2)
                raw_pose_t = pinn_out["pose"].squeeze(0)  # (4,)
                pinn_pose = {
                    "pitch": float(raw_pose_t[0].item()),
                    "yaw":   float(raw_pose_t[1].item()),
                    "roll":  float(raw_pose_t[2].item()),
                    "z":     float(raw_pose_t[3].item()),
                }
                # Apply slew-rate and absolute bounds via the optimizer.
                new_pose = self.optimizer.optimize(
                    predicted_temp=predicted_temp,
                    sensor_state=packet,
                    current_pose=self._current_pose,
                    pose_override=pinn_pose,
                )
            else:
                new_pose = self.optimizer.optimize(
                    predicted_temp=predicted_temp,
                    sensor_state=packet,
                    current_pose=self._current_pose,
                )
        except Exception as _pose_exc:  # noqa: BLE001
            _LOGGER.warning("pose extraction failed; using heuristic",
                            extra={"error": repr(_pose_exc)})
            new_pose = self.optimizer.optimize(
                predicted_temp=predicted_temp,
                sensor_state=packet,
                current_pose=self._current_pose,
            )

        # 7. Build the command.
        cmd = OrchestrationCommand(
            sim_mode=tier_name,
            aura_flag=True,
            pitch=float(new_pose["pitch"]),
            yaw=float(new_pose["yaw"]),
            roll=float(new_pose["roll"]),
            z=float(new_pose["z"]),
            predicted_temp=float(predicted_temp),
            uncertainty=float(uncertainty),
            fallback_active=False,
            timestamp=datetime.now(),
        )

        with self._state_lock:
            self._current_pose = new_pose
            self._last_command = cmd
            self._fallback_flags.append(False)
        return cmd

    # -- introspection --------------------------------------------- #

    def get_status(self) -> dict[str, Any]:
        """Return a snapshot of runtime metrics.

        Returns:
            Dict with keys:

            * ``n_steps`` — total steps observed in the rolling window.
            * ``mean_wall_s`` — mean ``step()`` wall time.
            * ``p99_wall_s`` — 99th-percentile ``step()`` wall time.
            * ``max_wall_s`` — max ``step()`` wall time.
            * ``fallback_rate`` — fraction of recent steps where
              ``fallback_active=True``.
            * ``consecutive_faults`` — current buffer fault counter.
            * ``last_solver_tier`` / ``last_solver_wall_s`` /
              ``last_solver_ok`` — dispatcher telemetry, or ``None``
              when dispatch is disabled or has not run.
        """
        with self._state_lock:
            walls = list(self._step_walls)
            flags = list(self._fallback_flags)
        n = len(walls)
        if n == 0:
            mean_wall = p99 = max_wall = 0.0
        else:
            mean_wall = sum(walls) / n
            ordered = sorted(walls)
            # Nearest-rank p99 so we never index OOB for small windows.
            idx = max(0, min(n - 1, int(round(0.99 * (n - 1)))))
            p99 = ordered[idx]
            max_wall = ordered[-1]
        fallback_rate = (sum(1 for f in flags if f) / len(flags)) if flags else 0.0
        last_tier = self.dispatcher.last_solver_tier if self.dispatcher else None
        last_wall = self.dispatcher.last_solver_wall_s if self.dispatcher else None
        last_ok = self.dispatcher.last_solver_ok if self.dispatcher else None
        return {
            "n_steps": n,
            "mean_wall_s": float(mean_wall),
            "p99_wall_s": float(p99),
            "max_wall_s": float(max_wall),
            "fallback_rate": float(fallback_rate),
            "consecutive_faults": int(self.buffer.fault_count()),
            "last_solver_tier": last_tier,
            "last_solver_wall_s": last_wall,
            "last_solver_ok": last_ok,
        }

    @property
    def current_pose(self) -> dict[str, float]:
        """Return a copy of the current pose dict."""
        with self._state_lock:
            return dict(self._current_pose)

    @property
    def last_command(self) -> OrchestrationCommand | None:
        """Return the most recent emitted command, or ``None`` on cold start."""
        with self._state_lock:
            return self._last_command
