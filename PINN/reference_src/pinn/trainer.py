"""Training loop for PINN-AURA-MFP (Batch C — Day 5 + Day 6).

This module implements the three-stage training schedule described in §6.2 of
the design doc:

1. :meth:`PinnTrainer.pretrain` — temperature + physics head only.
2. :meth:`PinnTrainer.sim_loop` — all components, routing loss warmed up.
3. :meth:`PinnTrainer.finetune` — joint optimisation with
   ``ReduceLROnPlateau``.

It also exposes:

* :func:`compute_loss` — the pure per-step loss function (Eq. 6.1).
* :func:`sample_collocation_points` — per-epoch physics collocation sampler.
* :func:`heuristic_route_label` — **DEPRECATED** legacy heuristic, kept
  as a baseline comparator; the main training path uses
  :func:`src.pinn.complexity.score_to_route_label`.
* :class:`CheckpointManager` — rolling-window ``.pt`` checkpoint writer.

All randomness is routed through :func:`src.utils.seeding.seed_everything` at
process start plus locally-seeded :class:`torch.Generator` / :class:`numpy.
random.Generator` instances; no module-level RNG state is mutated after
import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..config import PhysicsConfig, TrainingConfig
from ..utils.logging import get_logger
from .model import DualHeadPINN
from .physics import (
    celsius_to_kelvin,
    effective_irradiance,
    faiman_steady_state,
    physics_residual,
    wind_adjusted_tau,
)

__all__ = [
    "compute_loss",
    "sample_collocation_points",
    "heuristic_route_label",
    "PinnTrainer",
    "CheckpointManager",
]

_LOGGER = get_logger(__name__)

# Physical-domain ranges mirror ``SensorPacket.validate``'s contract. Kept
# private so changes to ``data.py`` do not silently widen the collocation
# domain.
_COLLOCATION_RANGES: dict[str, tuple[float, float]] = {
    "t_s": (0.0, 86400.0),
    "G_poa": (0.0, 1400.0),
    "T_amb": (-40.0, 70.0),
    "WS": (0.0, 60.0),
    "CC": (0.0, 1.0),
    "lat": (-90.0, 90.0),
    "lon": (-180.0, 180.0),
}

# Numeric-feature column order used by :class:`PinnDataset`.
_NUMERIC_ORDER: tuple[str, ...] = (
    "t_s",
    "G_poa",
    "T_amb",
    "WS",
    "CC",
    "lat",
    "lon",
)

# Maximum NaN/Inf occurrences tolerated per run before aborting.
_NAN_ABORT_THRESHOLD: int = 10

# Route index table (matches :attr:`ModelConfig.route_labels`).
_ROUTE_INDEX: dict[str, int] = {
    "LOFI": 0,
    "SIMV2": 1,
    "SIMV3": 2,
    "SIMV1": 3,
    "SIMV4": 4,
}


# ---------------------------------------------------------------------------
# Collocation sampling
# ---------------------------------------------------------------------------


def sample_collocation_points(
    n: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    image_size: tuple[int, int] = (32, 32),
) -> dict[str, Tensor]:
    """Uniformly sample ``n`` collocation points from the physical domain.

    The seven numeric features are drawn independently and uniformly from
    the same per-field ranges that :meth:`SensorPacket.validate` enforces.
    The ``t_s`` column is returned both inside the ``numeric`` tensor and
    as a separate leaf tensor ``t_s`` with ``requires_grad=True`` so the
    caller can compute ``dT/dt`` with :func:`physics_residual`. A tensor
    of zero-valued sky images is emitted so the sample can be fed to
    :class:`DualHeadPINN` directly.

    Args:
        n: Number of collocation points.
        device: Destination device for all tensors.
        generator: Optional :class:`torch.Generator` for deterministic
            sampling. A fresh unseeded generator is used if ``None``.
        image_size: Image spatial size matching :class:`SkyImageEncoder`.

    Returns:
        Dict with keys ``numeric (n, 7)``, ``image (n, 3, H, W)``, and
        scalar column references ``t_s``, ``T_amb``, ``G_poa``, ``WS``,
        ``CC`` each of shape ``(n, 1)``. ``t_s`` is a leaf requiring
        gradients; the others are detached helpers for convenience.
    """
    if n <= 0:
        raise ValueError(f"collocation sample size must be positive, got {n}")

    dev = torch.device(device) if not isinstance(device, torch.device) else device

    def _uniform(lo: float, hi: float) -> Tensor:
        # Sample on CPU with the supplied generator, then move to device so
        # the same generator produces identical draws regardless of where
        # the model lives.
        u = torch.rand((n, 1), generator=generator)
        return (lo + (hi - lo) * u).to(device=dev, dtype=torch.float32)

    t_s = _uniform(*_COLLOCATION_RANGES["t_s"]).clone().detach().requires_grad_(True)
    G_poa = _uniform(*_COLLOCATION_RANGES["G_poa"])
    T_amb = _uniform(*_COLLOCATION_RANGES["T_amb"])
    WS = _uniform(*_COLLOCATION_RANGES["WS"])
    CC = _uniform(*_COLLOCATION_RANGES["CC"])
    lat = _uniform(*_COLLOCATION_RANGES["lat"])
    lon = _uniform(*_COLLOCATION_RANGES["lon"])

    # Build numeric (n,7) with t_s as a differentiable column — the cat
    # keeps ``t_s`` in the autograd graph so grads back-propagate to it.
    numeric = torch.cat([t_s, G_poa, T_amb, WS, CC, lat, lon], dim=1)

    h, w = image_size
    image = torch.zeros((n, 3, h, w), dtype=torch.float32, device=dev)

    return {
        "numeric": numeric,
        "image": image,
        "t_s": t_s,
        "T_amb": T_amb,
        "G_poa": G_poa,
        "WS": WS,
        "CC": CC,
    }


# ---------------------------------------------------------------------------
# Heuristic routing label (DEPRECATED — Batch F replaced it with complexity
# scoring; kept for baseline comparison only).
# ---------------------------------------------------------------------------


def heuristic_route_label(packet: dict[str, Any]) -> int:
    """Legacy routing-label heuristic — **DEPRECATED baseline only**.

    Superseded by :func:`src.pinn.complexity.complexity_score` paired with
    :func:`src.pinn.complexity.score_to_route_label` (design doc §2.1 +
    §2.3). This function is retained only as a baseline comparator —
    :mod:`scripts.analyze_routing` contrasts its labels with the
    physics-derived ones to measure agreement. It is **not** called on
    the main training path.

    The implementation maps a record / packet dict to one of the route
    indices using the simple priority:

    1. ``CC > 0.8`` → SIMV1 (index 3) — heavy cloud, use image-only model.
    2. ``WS > 5`` or ``0.5 < CC <= 0.8`` → SIMV3 (index 2).
    3. ``G_poa < 100`` or dawn/dusk (``t_h < 6`` or ``t_h > 19``) → LOFI
       (index 0).
    4. Otherwise → SIMV2 (index 1).

    .. deprecated:: Batch F
       Use :func:`src.pinn.complexity.score_to_route_label` instead.

    Args:
        packet: Mapping with at least ``G_poa``, ``WS``, ``CC``, and
            either ``t_s`` (seconds) or ``t_h`` (hours).

    Returns:
        Integer route index in ``[0, 4]``.
    """
    CC = float(packet.get("CC", 0.0))
    WS = float(packet.get("WS", 0.0))
    G_poa = float(packet.get("G_poa", 0.0))
    if "t_h" in packet:
        t_h = float(packet["t_h"])
    else:
        t_h = float(packet.get("t_s", 0.0)) / 3600.0

    if CC > 0.8:
        return _ROUTE_INDEX["SIMV1"]
    if WS > 5.0 or CC > 0.5:
        return _ROUTE_INDEX["SIMV3"]
    if G_poa < 100.0 or t_h < 6.0 or t_h > 19.0:
        return _ROUTE_INDEX["LOFI"]
    return _ROUTE_INDEX["SIMV2"]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass
class _NaNGuard:
    """Shared mutable NaN counter passed into :func:`compute_loss`."""

    count: int = 0
    threshold: int = _NAN_ABORT_THRESHOLD


def _sanitize(term: Tensor, name: str, guard: _NaNGuard | None) -> Tensor:
    """Replace NaN/Inf with a zero constant and bump the NaN counter.

    The replacement is a fresh zero tensor detached from the graph; the
    corresponding loss term therefore contributes neither value nor
    gradient this step, but the run continues so the caller can detect
    chronic breakage via ``guard.count``.
    """
    if not torch.isfinite(term).all():
        if guard is not None:
            guard.count += 1
        print(
            f"[trainer] WARNING: non-finite value in loss term '{name}'",
            file=sys.stderr,
        )
        return torch.zeros((), dtype=term.dtype, device=term.device)
    return term


def compute_loss(
    model: DualHeadPINN,
    batch: dict[str, Tensor],
    collocation_inputs: dict[str, Tensor],
    training_cfg: TrainingConfig,
    *,
    route_weight_override: float | None = None,
    nan_guard: _NaNGuard | None = None,
) -> dict[str, Tensor]:
    """Compose the four-term PINN loss (Eq. 6.1).

    The returned dict always contains ``{"total", "data", "phys", "ic",
    "route"}``. Each term is guarded against non-finite values: if any
    term evaluates to NaN or Inf, the term is replaced with zero, a
    warning is written to ``stderr``, and ``nan_guard.count`` is
    incremented. Callers are expected to abort the run when the counter
    exceeds :data:`_NAN_ABORT_THRESHOLD`.

    Args:
        model: :class:`DualHeadPINN` with attached physics submodule.
        batch: Minibatch with keys ``numeric (B, 7)``, ``image (B, 3, H,
            W)``, ``T_panel (B, 1)`` in °C, and ``route_label (B, 1)``
            long.
        collocation_inputs: Output of :func:`sample_collocation_points`.
        training_cfg: Per-term weights (``lambda_*``).
        route_weight_override: If supplied, used in place of
            ``training_cfg.lambda_route`` — the sim-loop stage uses this
            to ramp the routing weight up from zero.
        nan_guard: Optional shared :class:`_NaNGuard` for the run-wide
            NaN counter.

    Returns:
        Dict of loss tensors; ``"total"`` is a differentiable scalar.
    """
    device = next(model.parameters()).device

    numeric = batch["numeric"].to(device)
    image = batch["image"].to(device)
    T_panel = batch["T_panel"].to(device)
    route_label = batch["route_label"].to(device).view(-1)

    # --- Data term -------------------------------------------------------
    out = model(numeric, image)
    T_hat = out["T_hat"]  # (B, 1) — interpreted as Celsius panel temp.
    route_logits = out["route_logits"]

    L_data = torch.mean((T_hat - T_panel) ** 2)

    # --- Routing term (weighted cross-entropy to prevent majority-class collapse) ---
    with torch.no_grad():
        counts = torch.bincount(route_label.long(), minlength=5).float().clamp(min=1.0)
        class_weights = (1.0 / counts)
        class_weights = class_weights / class_weights.sum() * 5.0  # normalise to mean=1
    L_route = nn.functional.cross_entropy(
        route_logits, route_label, weight=class_weights.to(device)
    )

    # --- Physics term (collocation points) ------------------------------
    coll_numeric = collocation_inputs["numeric"].to(device)
    coll_image = collocation_inputs["image"].to(device)
    coll_t_s = collocation_inputs["t_s"].to(device)
    coll_T_amb = collocation_inputs["T_amb"].to(device)
    coll_G_poa = collocation_inputs["G_poa"].to(device)
    coll_WS = collocation_inputs["WS"].to(device)
    coll_CC = collocation_inputs["CC"].to(device)

    coll_out = model(coll_numeric, coll_image)
    T_hat_coll = coll_out["T_hat"]

    phys = model.physics
    tau_0 = phys.tau_0
    U0 = phys.U0
    U1 = phys.U1
    gamma_CC = phys.gamma_CC

    M_spectral = torch.ones_like(coll_G_poa)
    G_eff = effective_irradiance(coll_G_poa, coll_CC, gamma_CC, M_spectral)
    T_ss_K = faiman_steady_state(
        celsius_to_kelvin(coll_T_amb), G_eff, U0, U1, coll_WS
    )
    T_ss = T_ss_K - 273.15  # compare in Celsius with T_hat
    tau_eff = wind_adjusted_tau(tau_0, U0, U1, coll_WS)

    residual = physics_residual(T_hat_coll, coll_t_s, T_ss, tau_eff)
    # Scale the residual so the dT/dt term and the (T_ss - T) term are of
    # comparable magnitude; t_s is in seconds so dT/dt is tiny. Divide by
    # a physically-meaningful reference time constant (tau_0 nominal) to
    # bring the residual into O(1). Detach the scale so we do not learn
    # toward a smaller ``tau_0`` via the loss shape.
    scale = tau_0.detach().clamp(min=1.0)
    L_phys = torch.mean((residual / scale) ** 2)

    # --- IC term (G=0, WS=0, t=0 -> T = T_amb) --------------------------
    # Use a small synthetic batch taken from the current minibatch's
    # ambient temperatures. Building it on-device keeps autograd happy.
    ic_numeric = torch.zeros_like(numeric)
    # index 2 = T_amb (per _NUMERIC_ORDER); copy from real batch
    ic_numeric[:, 2] = numeric[:, 2]
    ic_image = torch.zeros_like(image)
    ic_out = model(ic_numeric, ic_image)
    T_hat_ic = ic_out["T_hat"]
    T_amb_target = numeric[:, 2:3]  # (B, 1)
    L_ic = torch.mean((T_hat_ic - T_amb_target) ** 2)

    # --- Pose term (supervised when batch contains optimal_pose labels) ---
    # ``optimal_pose`` is a (B, 4) tensor [pitch_deg, yaw_deg, roll_deg, z_m].
    # It is only present once Phase 4 pose-sweep labels are generated.
    # Until then lambda_pose = 0 so this term contributes nothing.
    if "optimal_pose" in batch and hasattr(training_cfg, "lambda_pose") and training_cfg.lambda_pose > 0:
        optimal_pose = batch["optimal_pose"].to(device)  # (B, 4)
        pred_pose = out["pose"]                           # (B, 4)
        # Per-axis MSE, normalised so all four axes are comparably scaled.
        # Degrees: divide by typical range (~35, 180, 25 deg);
        # metres: divide by max z (3 m). These are soft normalisers.
        _scale = torch.tensor([35.0, 180.0, 25.0, 3.0], device=device)
        L_pose = torch.mean(((pred_pose - optimal_pose) / _scale) ** 2)
    else:
        L_pose = torch.zeros((), device=device)

    # --- Sanitize --------------------------------------------------------
    L_data = _sanitize(L_data, "data", nan_guard)
    L_phys = _sanitize(L_phys, "phys", nan_guard)
    L_ic = _sanitize(L_ic, "ic", nan_guard)
    L_route = _sanitize(L_route, "route", nan_guard)
    L_pose  = _sanitize(L_pose,  "pose",  nan_guard)

    lam_route = (
        training_cfg.lambda_route
        if route_weight_override is None
        else float(route_weight_override)
    )

    lam_pose = float(getattr(training_cfg, 'lambda_pose', 0.0))
    total = (
        training_cfg.lambda_data * L_data
        + training_cfg.lambda_phys * L_phys
        + training_cfg.lambda_IC * L_ic
        + lam_route * L_route
        + lam_pose * L_pose
    )

    return {
        "total": total,
        "data": L_data,
        "phys": L_phys,
        "ic": L_ic,
        "route": L_route,
        "pose": L_pose,
    }


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------


_EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


class CheckpointManager:
    """Rolling-window checkpoint writer with atomic saves + ``best.pt``.

    The last :attr:`keep_last` ``epoch_*.pt`` checkpoints are retained;
    older ones are deleted on each successful save. ``best.pt`` is
    maintained independently and overwritten whenever a new ``is_best``
    save arrives.
    """

    def __init__(self, directory: Path, keep_last: int = 5) -> None:
        """Create / reuse a checkpoint directory.

        Args:
            directory: Destination directory; created if missing.
            keep_last: Size of the rolling window. Must be positive.
        """
        if keep_last < 1:
            raise ValueError(f"keep_last must be >= 1, got {keep_last}")
        self.directory = Path(directory)
        self.keep_last = int(keep_last)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _epoch_paths(self) -> list[Path]:
        """Return existing ``epoch_*.pt`` files sorted ascending by epoch."""
        items: list[tuple[int, Path]] = []
        for child in self.directory.iterdir():
            m = _EPOCH_RE.match(child.name)
            if m:
                items.append((int(m.group(1)), child))
        items.sort(key=lambda t: t[0])
        return [p for _, p in items]

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metrics: dict[str, Any],
        is_best: bool,
    ) -> Path:
        """Atomically save ``state`` to ``epoch_<epoch>.pt`` (and best.pt).

        Args:
            state: Dict produced by :meth:`PinnTrainer._snapshot`.
            epoch: Epoch index (1-based).
            metrics: JSON-serializable metrics to embed in the payload.
            is_best: If ``True``, also overwrites ``best.pt``.

        Returns:
            Path of the primary ``epoch_<epoch>.pt`` file written.
        """
        payload = {
            "epoch": int(epoch),
            "metrics": metrics,
            **state,
        }
        dest = self.directory / f"epoch_{int(epoch)}.pt"
        self._atomic_torch_save(payload, dest)
        if is_best:
            best = self.directory / "best.pt"
            self._atomic_torch_save(payload, best)

        # Rolling-window pruning.
        paths = self._epoch_paths()
        if len(paths) > self.keep_last:
            for old in paths[: -self.keep_last]:
                try:
                    old.unlink()
                except FileNotFoundError:  # pragma: no cover - race
                    pass
        return dest

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], dest: Path) -> None:
        """``torch.save`` to ``<dest>.tmp`` then ``os.replace`` into place."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, dest)

    def load(self, path: Path) -> dict[str, Any]:
        """Load a checkpoint dict.

        Args:
            path: Source path. May be any file written by :meth:`save`.

        Returns:
            Raw checkpoint dict with keys ``epoch, metrics, model_state,
            optimizer_state, scheduler_state, rng_state``.
        """
        src = Path(path)
        # weights_only=False: checkpoints include optimizer + RNG state
        # which are not pure tensors. This code only loads checkpoints the
        # trainer itself produced (same directory, same machine).
        return torch.load(src, map_location="cpu", weights_only=False)  # type: ignore[no-any-return]

    def resume(
        self,
        trainer: "PinnTrainer",
        path: Path,
    ) -> int:
        """Restore ``trainer`` state from ``path``.

        Args:
            trainer: The :class:`PinnTrainer` to mutate.
            path: Checkpoint file.

        Returns:
            The epoch index recorded in the checkpoint.
        """
        ck = self.load(path)
        trainer.model.load_state_dict(ck["model_state"])
        if ck.get("optimizer_state") is not None:
            trainer.optimizer.load_state_dict(ck["optimizer_state"])
        if ck.get("scheduler_state") is not None and trainer.scheduler is not None:
            trainer.scheduler.load_state_dict(ck["scheduler_state"])
        rng = ck.get("rng_state")
        if rng is not None:
            _restore_rng(rng)
        trainer._resumed_epoch = int(ck.get("epoch", 0))
        return trainer._resumed_epoch


def _capture_rng() -> dict[str, Any]:
    """Snapshot Python, NumPy, and Torch RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state: dict[str, Any]) -> None:
    """Inverse of :func:`_capture_rng`.

    ``random.getstate`` returns a tuple, but some serialization paths (for
    example, JSON or YAML round-trips) coerce it to a list. Coerce back to
    tuple so :func:`random.setstate` is happy in either case.
    """
    if "python" in state:
        py_state = state["python"]
        if isinstance(py_state, list):
            py_state = tuple(py_state)
        random.setstate(py_state)
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PinnTrainer:
    """Three-stage trainer composing pretrain → sim-loop → finetune.

    Parameters are weight-decayed only through the optimizer passed in;
    this class does not rewrap it. The constructor does not seed global
    RNGs — callers must invoke :func:`src.utils.seeding.seed_everything`
    ahead of time.
    """

    def __init__(
        self,
        model: DualHeadPINN,
        optimizer: Optimizer,
        scheduler: Any | None,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        training_cfg: TrainingConfig,
        physics_cfg: PhysicsConfig,
        device: torch.device | str = "cpu",
        logger: Any | None = None,
        *,
        checkpoint_dir: Path | None = None,
        log_dir: Path | None = None,
        checkpoint_every: int = 50,
        early_stop: bool = False,
        early_stop_patience: int = 100,
        collocation_points: int | None = None,
        grad_clip: float | None = None,
    ) -> None:
        """Wire up the three-stage training schedule."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.training_cfg = training_cfg
        self.physics_cfg = physics_cfg
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.logger = logger if logger is not None else _LOGGER

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path("checkpoints")
        self.log_dir = Path(log_dir) if log_dir is not None else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt = CheckpointManager(self.checkpoint_dir, keep_last=5)

        self.checkpoint_every = int(checkpoint_every)
        self.early_stop = bool(early_stop)
        self.early_stop_patience = int(early_stop_patience)
        self.grad_clip = float(grad_clip) if grad_clip is not None else float(training_cfg.grad_clip)
        self.collocation_n = int(
            collocation_points if collocation_points is not None else training_cfg.collocation_points
        )

        self.model.to(self.device)

        # NaN guard shared across stages.
        self._nan_guard = _NaNGuard(count=0, threshold=_NAN_ABORT_THRESHOLD)

        # Collocation RNG — seeded deterministically per epoch from the
        # CPU RNG to respect ``seed_everything``.
        self._collocation_seed_base = int(
            torch.randint(0, 2**31 - 1, (1,)).item()
        )

        # Timestamp used for log filenames; same for all stages in a run.
        self._run_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.jsonl_path = self.log_dir / f"training_{self._run_stamp}.jsonl"
        # Truncate on start so resumed logs get a fresh file.
        self.jsonl_path.write_text("", encoding="utf-8")

        # Per-stage accumulated rows for plotting.
        self._stage_rows: list[dict[str, Any]] = []

        # Tracking state.
        self.global_epoch: int = 0
        self.best_val_total: float = float("inf")
        self._resumed_epoch: int = 0

    # ------------------------------------------------------------------
    # Public stage methods
    # ------------------------------------------------------------------

    def pretrain(self, epochs: int) -> dict[str, Any]:
        """Stage 1 — temperature head + physics parameters only."""
        self._set_route_head_trainable(False)
        self._stage_rows = []
        history = self._run_epochs(
            epochs=epochs,
            stage="pretrain",
            route_weight_fn=lambda ep: 0.0,
        )
        # Convergence check on the last 20 phys losses.
        phys_tail = [r["train_L_phys"] for r in self._stage_rows[-20:]]
        if phys_tail:
            tail_mean = float(np.mean(phys_tail))
            history["phys_tail_mean"] = tail_mean
            if tail_mean < 0.05:
                self.logger.info(
                    "pretrain converged",
                    extra={"phys_tail_mean": tail_mean},
                )
            else:
                self.logger.warning(
                    "pretrain did not converge",
                    extra={"phys_tail_mean": tail_mean},
                )
        self._render_stage_plot("pretrain")
        # Re-enable routing head for subsequent stages.
        self._set_route_head_trainable(True)
        return history

    def sim_loop(self, epochs: int, route_warmup: int) -> dict[str, Any]:
        """Stage 2 — all components; routing loss ramps in after warmup."""
        self._set_route_head_trainable(True)
        self._stage_rows = []
        ramp_epochs = 50
        target = float(self.training_cfg.lambda_route)

        def _route_weight(stage_epoch: int) -> float:
            # stage_epoch is 1-based within the stage.
            if stage_epoch <= route_warmup:
                return 0.0
            ramped = stage_epoch - route_warmup
            if ramped >= ramp_epochs:
                return target
            return target * (ramped / ramp_epochs)

        history = self._run_epochs(
            epochs=epochs,
            stage="sim_loop",
            route_weight_fn=_route_weight,
        )
        self._render_stage_plot("sim_loop")
        return history

    def finetune(self, epochs: int) -> dict[str, Any]:
        """Stage 3 — joint optimisation with ``ReduceLROnPlateau``."""
        self._set_route_head_trainable(True)
        # Re-bind the scheduler so finetune always owns a plateau scheduler
        # on validation total loss.
        plateau = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.training_cfg.lr_scheduler_factor,
            patience=self.training_cfg.lr_scheduler_patience,
        )
        prev_scheduler = self.scheduler
        self.scheduler = plateau
        self._stage_rows = []
        try:
            history = self._run_epochs(
                epochs=epochs,
                stage="finetune",
                route_weight_fn=lambda ep: float(self.training_cfg.lambda_route),
                plateau_on_val=True,
            )
        finally:
            self.scheduler = prev_scheduler
        self._render_stage_plot("finetune")
        return history

    def train(self) -> dict[str, dict[str, Any]]:
        """Default entry point running all three stages sequentially."""
        hist_pre = self.pretrain(self.training_cfg.pretrain_epochs)
        hist_sim = self.sim_loop(
            self.training_cfg.sim_loop_epochs,
            self.training_cfg.route_warmup_epochs,
        )
        hist_ft = self.finetune(self.training_cfg.finetune_epochs)
        return {"pretrain": hist_pre, "sim_loop": hist_sim, "finetune": hist_ft}

    # ------------------------------------------------------------------
    # Core epoch loop
    # ------------------------------------------------------------------

    def _run_epochs(
        self,
        epochs: int,
        stage: str,
        route_weight_fn: Callable[[int], float],
        plateau_on_val: bool = False,
    ) -> dict[str, Any]:
        """Run ``epochs`` epochs in the given stage."""
        if epochs <= 0:
            return {"epochs": 0}

        best_metric = float("inf")
        patience_counter = 0
        start_wall = time.time()

        for local_epoch in range(1, epochs + 1):
            self.global_epoch += 1
            lam_route = float(route_weight_fn(local_epoch))

            # Re-sample collocation points *every epoch*.
            gen = torch.Generator()
            gen.manual_seed(self._collocation_seed_base + self.global_epoch)
            coll = sample_collocation_points(
                self.collocation_n, device=self.device, generator=gen
            )

            train_metrics = self._train_one_epoch(coll, lam_route)
            val_metrics = self._evaluate(coll, lam_route)

            if self._nan_guard.count > self._nan_guard.threshold:
                raise RuntimeError(
                    f"training aborted: NaN counter exceeded "
                    f"{self._nan_guard.threshold} "
                    f"(stage={stage}, epoch={self.global_epoch})"
                )

            lr = float(self.optimizer.param_groups[0]["lr"])
            wall = time.time() - start_wall
            route_head_grad_norm = train_metrics.pop("route_head_grad_norm", 0.0)

            row: dict[str, Any] = {
                "epoch": self.global_epoch,
                "stage_epoch": local_epoch,
                "stage": stage,
                "wall_time_s": round(wall, 4),
                "lr": lr,
                "route_weight": lam_route,
                "route_head_grad_norm": route_head_grad_norm,
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                **self._physics_snapshot(),
            }
            self._stage_rows.append(row)
            self._append_jsonl(row)

            self.logger.info(
                f"epoch {self.global_epoch} ({stage})",
                extra={
                    "stage": stage,
                    "epoch": self.global_epoch,
                    "train_total": row["train_L_total"],
                    "val_total": row["val_L_total"],
                    "lr": lr,
                },
            )

            # Scheduler step.
            if plateau_on_val and self.scheduler is not None:
                # Some schedulers need the metric (plateau) vs none (others).
                try:
                    self.scheduler.step(row["val_L_total"])
                except TypeError:  # pragma: no cover - defensive
                    self.scheduler.step()
            elif self.scheduler is not None:
                try:
                    self.scheduler.step()
                except TypeError:  # plateau fallback without metric
                    pass

            # Best/checkpoint handling.
            is_best = row["val_L_total"] < self.best_val_total
            if is_best:
                self.best_val_total = row["val_L_total"]

            save_now = (
                self.global_epoch % max(1, self.checkpoint_every) == 0 or is_best
            )
            if save_now:
                self.ckpt.save(
                    self._snapshot(),
                    epoch=self.global_epoch,
                    metrics={
                        "train_L_total": row["train_L_total"],
                        "val_L_total": row["val_L_total"],
                        "stage": stage,
                    },
                    is_best=is_best,
                )

            # Early stopping (opt-in only).
            if self.early_stop:
                if row["val_L_total"] < best_metric - 1e-6:
                    best_metric = row["val_L_total"]
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stop_patience:
                        self.logger.info(
                            "early stop triggered",
                            extra={"stage": stage, "epoch": self.global_epoch},
                        )
                        break

        return {
            "epochs": epochs,
            "last_train_total": self._stage_rows[-1]["train_L_total"]
            if self._stage_rows
            else None,
            "last_val_total": self._stage_rows[-1]["val_L_total"]
            if self._stage_rows
            else None,
        }

    def _train_one_epoch(
        self,
        coll: dict[str, Tensor],
        lam_route: float,
    ) -> dict[str, float]:
        """One optimizer pass over ``train_loader``."""
        self.model.train()
        sums = {"L_total": 0.0, "L_data": 0.0, "L_phys": 0.0, "L_IC": 0.0, "L_route": 0.0, "L_pose": 0.0}
        n_batches = 0
        n_correct = 0
        n_total = 0
        route_head_grad_sq = 0.0
        route_head_grad_batches = 0

        for batch in self.train_loader:
            losses = compute_loss(
                self.model,
                batch,
                coll,
                self.training_cfg,
                route_weight_override=lam_route,
                nan_guard=self._nan_guard,
            )
            self.optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()  # type: ignore[no-untyped-call]
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
            # Measure routing-head gradient norm (pre-step, post-backward).
            rh_sq = 0.0
            for p in self.model.route_head.parameters():
                if p.grad is not None:
                    rh_sq += float(p.grad.detach().pow(2).sum().item())
            route_head_grad_sq += rh_sq
            route_head_grad_batches += 1

            self.optimizer.step()

            sums["L_total"] += float(losses["total"].detach().item())
            sums["L_data"] += float(losses["data"].detach().item())
            sums["L_phys"] += float(losses["phys"].detach().item())
            sums["L_IC"] += float(losses["ic"].detach().item())
            sums["L_route"] += float(losses["route"].detach().item())
            sums["L_pose"]  += float(losses["pose"].detach().item())

            with torch.no_grad():
                numeric = batch["numeric"].to(self.device)
                image = batch["image"].to(self.device)
                logits = self.model(numeric, image)["route_logits"]
                preds = torch.argmax(logits, dim=-1)
                n_correct += int((preds == batch["route_label"].to(self.device).view(-1)).sum().item())
                n_total += int(preds.numel())
            n_batches += 1

        if n_batches == 0:
            raise RuntimeError("train_loader produced zero batches")
        result = {k: v / n_batches for k, v in sums.items()}
        result["route_accuracy"] = n_correct / max(1, n_total)
        avg_rh = math.sqrt(route_head_grad_sq / max(1, route_head_grad_batches))
        result["route_head_grad_norm"] = avg_rh
        return result

    def _evaluate(
        self,
        coll: dict[str, Tensor],
        lam_route: float,
    ) -> dict[str, float]:
        """Evaluate on ``val_loader``; returns per-term means + routing acc."""
        self.model.eval()
        sums = {"L_total": 0.0, "L_data": 0.0, "L_phys": 0.0, "L_IC": 0.0, "L_route": 0.0, "L_pose": 0.0}
        n_batches = 0
        n_correct = 0
        n_total = 0
        unc_sum = 0.0
        unc_count = 0

        # NOTE: physics loss requires autograd on the collocation inputs,
        # so we cannot wrap the entire eval in ``torch.no_grad``. We
        # instead manually avoid backward + optimizer steps.
        for batch in self.val_loader:
            losses = compute_loss(
                self.model,
                batch,
                coll,
                self.training_cfg,
                route_weight_override=lam_route,
                nan_guard=self._nan_guard,
            )
            sums["L_total"] += float(losses["total"].detach().item())
            sums["L_data"] += float(losses["data"].detach().item())
            sums["L_phys"] += float(losses["phys"].detach().item())
            sums["L_IC"] += float(losses["ic"].detach().item())
            sums["L_route"] += float(losses["route"].detach().item())
            sums["L_pose"]  += float(losses["pose"].detach().item())

            with torch.no_grad():
                numeric = batch["numeric"].to(self.device)
                image = batch["image"].to(self.device)
                out = self.model.predict_with_uncertainty(numeric, image)
                preds = torch.argmax(out["route_probs"], dim=-1)
                n_correct += int(
                    (preds == batch["route_label"].to(self.device).view(-1)).sum().item()
                )
                n_total += int(preds.numel())
                unc_sum += float(out["route_uncertainty"].sum().item())
                unc_count += int(out["route_uncertainty"].numel())
            n_batches += 1

        if n_batches == 0:
            # Empty val loader — still emit zeros to keep logging shape stable.
            zero = {k: 0.0 for k in sums}
            zero["route_accuracy"] = 0.0
            zero["route_uncertainty_mean"] = 0.0
            return zero
        result = {k: v / n_batches for k, v in sums.items()}
        result["route_accuracy"] = n_correct / max(1, n_total)
        result["route_uncertainty_mean"] = unc_sum / max(1, unc_count)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_route_head_trainable(self, flag: bool) -> None:
        """Toggle ``requires_grad`` on the routing head parameters."""
        for p in self.model.route_head.parameters():
            p.requires_grad_(flag)

    def _physics_snapshot(self) -> dict[str, float]:
        """Current exponentiated physics parameters as plain floats."""
        phys = self.model.physics
        return {
            "tau_0": float(phys.tau_0.detach().item()),
            "U0": float(phys.U0.detach().item()),
            "U1": float(phys.U1.detach().item()),
            "gamma_CC": float(phys.gamma_CC.detach().item()),
        }

    def _snapshot(self) -> dict[str, Any]:
        """Snapshot model + optimizer + scheduler + RNG state for checkpoints."""
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "rng_state": _capture_rng(),
            "best_val_total": self.best_val_total,
            "global_epoch": self.global_epoch,
        }

    def _append_jsonl(self, row: dict[str, Any]) -> None:
        """Append one JSON-lines record to the per-run log file."""
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def _render_stage_plot(self, stage: str) -> None:
        """Save a 4-subplot PNG summarising the stage just completed."""
        if not self._stage_rows:
            return
        try:
            import matplotlib  # noqa: WPS433

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: WPS433
        except ImportError:  # pragma: no cover - matplotlib in requirements
            return

        epochs = [r["epoch"] for r in self._stage_rows]
        train_total = [r["train_L_total"] for r in self._stage_rows]
        val_total = [r["val_L_total"] for r in self._stage_rows]
        train_data = [r["train_L_data"] for r in self._stage_rows]
        train_phys = [r["train_L_phys"] for r in self._stage_rows]
        train_ic = [r["train_L_IC"] for r in self._stage_rows]
        train_route = [r["train_L_route"] for r in self._stage_rows]
        tau0 = [r["tau_0"] for r in self._stage_rows]
        U0 = [r["U0"] for r in self._stage_rows]
        U1 = [r["U1"] for r in self._stage_rows]
        gCC = [r["gamma_CC"] for r in self._stage_rows]
        acc = [r["train_route_accuracy"] for r in self._stage_rows]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(epochs, train_total, label="train")
        axes[0, 0].plot(epochs, val_total, label="val")
        axes[0, 0].set_title(f"Total loss — {stage}")
        axes[0, 0].set_yscale("log")
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, train_data, label="data")
        axes[0, 1].plot(epochs, train_phys, label="phys")
        axes[0, 1].plot(epochs, train_ic, label="ic")
        axes[0, 1].plot(epochs, train_route, label="route")
        axes[0, 1].set_title("Per-term (train)")
        axes[0, 1].set_yscale("log")
        axes[0, 1].legend()

        axes[1, 0].plot(epochs, tau0, label="tau_0")
        axes[1, 0].plot(epochs, U0, label="U0")
        axes[1, 0].plot(epochs, U1, label="U1")
        axes[1, 0].plot(epochs, gCC, label="gamma_CC")
        axes[1, 0].set_title("Learned physics params")
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, acc, label="train acc")
        axes[1, 1].set_title("Routing accuracy")
        axes[1, 1].set_ylim(0.0, 1.0)

        fig.tight_layout()
        out = self.log_dir / f"training_curves_{self._run_stamp}_{stage}.png"
        fig.savefig(out, dpi=90)
        plt.close(fig)

    # Public helper for tests / external code -------------------------------

    def state_dict_hash(self) -> str:
        """Stable hash of the model's ``state_dict`` (for determinism tests)."""
        m = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            m.update(name.encode("utf-8"))
            m.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return m.hexdigest()
