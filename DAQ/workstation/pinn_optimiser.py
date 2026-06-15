"""workstation/pinn_optimiser.py – PINN-based PSO optimiser.

Uses a trained PINN as a differentiable surrogate to find the (tilt, azimuth,
height) triplet that maximises the predicted maximum power P_mp.

When the PINN checkpoint is unavailable (e.g. in unit tests or CI) the
optimiser falls back to a lightweight analytic surrogate so the rest of the
pipeline can be exercised without a GPU or the full PINN-AURA-MFP repo.

PSO implementation
------------------
A vanilla Particle Swarm Optimiser is used.  The swarm position is a 2-D
array of shape ``(n_particles, 3)`` where the three columns correspond to
[tilt, azimuth, height].  The objective is **maximisation** of P_mp as
predicted by the PINN (or the fallback surrogate).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("workstation.pinn_optimiser")

# ---------------------------------------------------------------------------
# PSO hyper-parameters (used when not overridden by the caller)
# ---------------------------------------------------------------------------
_DEFAULT_N_PARTICLES: int = 50
_DEFAULT_N_ITERATIONS: int = 100
_W: float = 0.7298   # inertia weight
_C1: float = 1.4962  # cognitive coefficient
_C2: float = 1.4962  # social coefficient


class PINNOptimiser:
    """Find optimal panel configuration using a PINN surrogate + PSO.

    Parameters
    ----------
    pinn_root:
        Path to the ``PINN-AURA-MFP`` repository root.  Used to locate the
        PINN inference module when ``use_pinn=True``.
    checkpoint_path:
        Path to the trained PyTorch checkpoint (``*.pt`` file).
    use_pinn:
        When ``True`` (default), attempt to load the PINN checkpoint.  When
        ``False``, or when loading fails, the analytic surrogate is used.
    """

    def __init__(
        self,
        pinn_root: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        *,
        use_pinn: bool = True,
    ) -> None:
        self._pinn_root = Path(pinn_root) if pinn_root else None
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._model = None

        if use_pinn:
            self._model = self._load_pinn()

    # ── Public API ────────────────────────────────────────────────────────────

    def optimise(
        self,
        packet: dict[str, Any],
        *,
        n_particles: int = _DEFAULT_N_PARTICLES,
        n_iterations: int = _DEFAULT_N_ITERATIONS,
        tilt_range: tuple[float, float] = (0.0, 90.0),
        azimuth_range: tuple[float, float] = (-180.0, 180.0),
        height_range: tuple[float, float] = (0.5, 5.0),
    ) -> dict[str, Any]:
        """Run PSO to find the optimal (tilt, azimuth, height).

        Parameters
        ----------
        packet:
            Current sensor packet (``PINN_SENSOR_PACKET_SCHEMA``).
        n_particles:
            Swarm size (default: 50).
        n_iterations:
            Maximum number of PSO iterations (default: 100).
        tilt_range, azimuth_range, height_range:
            Search bounds for the three decision variables.

        Returns
        -------
        dict
            Keys: ``tilt_opt``, ``azimuth_opt``, ``height_opt``,
            ``P_mp_pred``, ``T_panel_pred``, ``n_iterations``.
        """
        lb = np.array([tilt_range[0], azimuth_range[0], height_range[0]], dtype=float)
        ub = np.array([tilt_range[1], azimuth_range[1], height_range[1]], dtype=float)

        # ── Initialise swarm ──────────────────────────────────────────────
        rng = np.random.default_rng()
        pos = rng.uniform(lb, ub, size=(n_particles, 3))
        vel = rng.uniform(-(ub - lb), ub - lb, size=(n_particles, 3))

        p_best_pos = pos.copy()
        p_best_val = np.full(n_particles, -np.inf)

        g_best_pos = pos[0].copy()
        g_best_val = -np.inf

        # ── Evaluate initial positions ────────────────────────────────────
        for i in range(n_particles):
            val = self._evaluate(pos[i], packet)
            p_best_val[i] = val
            p_best_pos[i] = pos[i].copy()
            if val > g_best_val:
                g_best_val = val
                g_best_pos = pos[i].copy()

        # ── Main PSO loop ─────────────────────────────────────────────────
        for _ in range(n_iterations):
            r1 = rng.random((n_particles, 3))
            r2 = rng.random((n_particles, 3))

            vel = (
                _W * vel
                + _C1 * r1 * (p_best_pos - pos)
                + _C2 * r2 * (g_best_pos - pos)
            )
            pos = np.clip(pos + vel, lb, ub)

            for i in range(n_particles):
                val = self._evaluate(pos[i], packet)
                if val > p_best_val[i]:
                    p_best_val[i] = val
                    p_best_pos[i] = pos[i].copy()
                if val > g_best_val:
                    g_best_val = val
                    g_best_pos = pos[i].copy()

        tilt_opt, az_opt, h_opt = float(g_best_pos[0]), float(g_best_pos[1]), float(g_best_pos[2])
        t_panel_pred = self._predict_t_panel(tilt_opt, packet)

        return {
            "tilt_opt": tilt_opt,
            "azimuth_opt": az_opt,
            "height_opt": h_opt,
            "P_mp_pred": float(g_best_val),
            "T_panel_pred": t_panel_pred,
            "n_iterations": n_iterations,
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_pinn(self) -> Any:
        """Attempt to load the PINN model from the checkpoint.

        Returns the loaded model, or ``None`` when loading fails (which causes
        the fallback analytic surrogate to be used).
        """
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            log.debug(
                "pinn_optimiser: checkpoint not found at %s — using analytic surrogate",
                self._checkpoint_path,
            )
            return None
        try:
            self._ensure_pinn_root_on_path()

            import torch  # noqa: PLC0415

            model = torch.load(str(self._checkpoint_path), map_location="cpu")
            model.eval()
            log.info(
                "pinn_optimiser: loaded PINN checkpoint from %s",
                self._checkpoint_path,
            )
            return model
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "pinn_optimiser: failed to load PINN — falling back to analytic surrogate (%s)",
                exc,
            )
            return None

    def _ensure_pinn_root_on_path(self) -> None:
        """Insert pinn_root at the front of sys.path when not already present."""
        if self._pinn_root is None:
            return
        import sys  # noqa: PLC0415

        pinn_str = str(self._pinn_root)
        if pinn_str not in sys.path:
            sys.path.insert(0, pinn_str)

    def _evaluate(
        self,
        position: np.ndarray,
        packet: dict[str, Any],
    ) -> float:
        """Evaluate P_mp for a given (tilt, azimuth, height) position.

        Delegates to the PINN when available; otherwise uses the analytic
        surrogate.
        """
        tilt, azimuth, height = position
        if self._model is not None:
            return self._pinn_evaluate(tilt, azimuth, height, packet)
        return self._analytic_surrogate(tilt, azimuth, height, packet)

    def _pinn_evaluate(
        self,
        tilt: float,
        azimuth: float,
        height: float,
        packet: dict[str, Any],
    ) -> float:
        """Query the loaded PINN model for P_mp.

        Constructs a feature vector matching the PINN input schema and
        returns the scalar P_mp prediction.
        """
        try:
            import torch  # noqa: PLC0415

            g_poa = float(packet.get("G_poa") or 0.0)
            t_amb = float(packet.get("T_amb") or 25.0)
            ws = float(packet.get("WS") or 0.0)
            x = torch.tensor(
                [[g_poa, t_amb, ws, tilt, azimuth, height]],
                dtype=torch.float32,
            )
            with torch.no_grad():
                out = self._model(x)
            return float(out.squeeze())
        except Exception as exc:  # noqa: BLE001
            log.debug("pinn_optimiser: PINN inference failed — using surrogate (%s)", exc)
            return self._analytic_surrogate(tilt, azimuth, height, packet)

    @staticmethod
    def _analytic_surrogate(
        tilt: float,
        azimuth: float,
        height: float,
        packet: dict[str, Any],
    ) -> float:
        """Lightweight analytic surrogate for P_mp.

        Used in CI and when the PINN checkpoint is unavailable.  The model
        captures the main physical trends:
        - P_mp ∝ G_poa × cos(tilt - optimal_tilt)
        - azimuth penalty (prefer south-facing ~0°)
        - height has a small positive effect up to ~2 m (mounting height
          trades off wind cooling vs. structural loads)

        This surrogate is intentionally simple and is **not** used in
        production inference.
        """
        import math  # noqa: PLC0415

        g_poa = float(packet.get("G_poa") or 0.0)
        t_amb = float(packet.get("T_amb") or 25.0)

        # Latitude-dependent optimal tilt (fall back to 35° if lat missing)
        lat = abs(float(packet.get("lat") or 35.0))
        optimal_tilt = lat * 0.76 + 3.1  # empirical approximation

        tilt_factor = math.cos(math.radians(tilt - optimal_tilt))
        az_factor = math.cos(math.radians(azimuth)) * 0.05 + 0.95
        h_factor = 1.0 + 0.02 * min(height, 2.0)

        # Temperature derating (≈ -0.45 %/°C above 25 °C for crystalline Si)
        t_factor = 1.0 - 0.0045 * max(t_amb - 25.0, 0.0)

        p_mp = g_poa * 0.20 * 1.6 * tilt_factor * az_factor * h_factor * t_factor
        return max(p_mp, 0.0)

    @staticmethod
    def _predict_t_panel(tilt: float, packet: dict[str, Any]) -> float:
        """Estimate panel temperature at the optimal tilt.

        Uses the Faiman thermal model as a lightweight approximation.
        """
        g_poa = float(packet.get("G_poa") or 0.0)
        t_amb = float(packet.get("T_amb") or 25.0)
        ws = float(packet.get("WS") or 1.0)

        # Faiman coefficients (crystalline Si, free-standing)
        u0, u1 = 25.0, 6.84
        t_panel = t_amb + g_poa / (u0 + u1 * max(ws, 0.1))
        return float(t_panel)
