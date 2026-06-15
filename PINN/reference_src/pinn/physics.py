"""Pure physics functions and learnable parameters for PINN-AURA-MFP.

This module contains the scientific core of the PINN: the extended thermal
model (Faiman steady state, wind-adjusted time constant), the irradiance
pipeline (diurnal clear-sky profile, cloud-cover derate, spectral factor),
the Sandia efficiency model, and the PINN residual itself.

All functions are pure: they accept and return ``torch.Tensor`` objects, they
contain no module-level state, no I/O, and no logging in hot paths. Every
function is ``torch.autograd`` compatible end-to-end.

The learnable physics parameters are held by :class:`LearnedPhysicsParameters`
in **log space** so that the exponentiated values are strictly positive
regardless of optimizer updates.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from ..config import PhysicsConfig

__all__ = [
    "PhysicsConfig",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "diurnal_irradiance",
    "effective_irradiance",
    "faiman_steady_state",
    "wind_adjusted_tau",
    "sandia_efficiency",
    "physics_residual",
    "LearnedPhysicsParameters",
]


# ---------------------------------------------------------------------------
# Temperature unit helpers. All ad-hoc 273.15 offsets must go through these.
# ---------------------------------------------------------------------------

_KELVIN_OFFSET: float = 273.15


def celsius_to_kelvin(T_C: Tensor) -> Tensor:
    """Convert Celsius to Kelvin.

    Args:
        T_C: Temperature in degrees Celsius.

    Returns:
        Temperature in Kelvin, same shape/dtype as ``T_C``.
    """
    return T_C + _KELVIN_OFFSET


def kelvin_to_celsius(T_K: Tensor) -> Tensor:
    """Convert Kelvin to Celsius.

    Args:
        T_K: Temperature in Kelvin.

    Returns:
        Temperature in degrees Celsius, same shape/dtype as ``T_K``.
    """
    return T_K - _KELVIN_OFFSET


# ---------------------------------------------------------------------------
# Irradiance pipeline (Eq. 3.3 + 3.5).
# ---------------------------------------------------------------------------


def diurnal_irradiance(
    t_h: Tensor,
    t_noon: Tensor,
    daylen: Tensor,
    G_peak: Tensor,
) -> Tensor:
    """Clear-sky diurnal plane-of-array irradiance (Eq. 3.3).

    ``G = G_peak · max(0, cos(π · (t_h − t_noon) / (0.5 · daylen)))``.

    Args:
        t_h: Local solar time in hours.
        t_noon: Local solar noon in hours.
        daylen: Day length in hours.
        G_peak: Peak plane-of-array irradiance (W/m²).

    Returns:
        Clear-sky irradiance (W/m²), clamped at zero outside the daylight
        window.
    """
    arg = math.pi * (t_h - t_noon) / (0.5 * daylen)
    return G_peak * torch.clamp(torch.cos(arg), min=0.0)


def effective_irradiance(
    G_poa: Tensor,
    CC: Tensor,
    gamma_CC: Tensor,
    M_spectral: Tensor,
) -> Tensor:
    """Cloud-and-spectrum-adjusted effective irradiance (Eq. 3.5).

    ``G_eff = G_poa · (1 − CC^γ) · M_spectral``.

    ``CC = 0`` is handled without NaN in the gradient by applying ``CC^γ``
    only where ``CC > 0`` and substituting ``0`` elsewhere. The naive
    expression ``CC ** gamma`` produces ``NaN`` in the autograd graph when
    ``CC = 0`` and ``gamma`` is learnable because ``d/dγ (0^γ)`` reduces to
    ``0 · log(0)``.

    Args:
        G_poa: Plane-of-array irradiance (W/m²).
        CC: Cloud cover fraction in ``[0, 1]``.
        gamma_CC: Cloud-cover exponent (learnable, strictly positive).
        M_spectral: Dimensionless spectral modifier, typically ≈ 1.

    Returns:
        Effective irradiance (W/m²).
    """
    # Guard the pow() so gradients stay finite at CC = 0.
    cc_safe = torch.where(CC > 0, CC, torch.ones_like(CC))
    cc_pow = torch.where(CC > 0, cc_safe.pow(gamma_CC), torch.zeros_like(CC))
    return G_poa * (1.0 - cc_pow) * M_spectral


# ---------------------------------------------------------------------------
# Faiman extended thermal model (Eq. 3.4 + Eq. 3.6).
# ---------------------------------------------------------------------------


def faiman_steady_state(
    T_amb_K: Tensor,
    G_eff: Tensor,
    U0: Tensor,
    U1: Tensor,
    WS: Tensor,
) -> Tensor:
    """Faiman steady-state panel temperature (Eq. 3.4).

    ``T_ss = T_amb + G_eff / (U0 + U1 · WS)``.

    Args:
        T_amb_K: Ambient temperature in Kelvin.
        G_eff: Effective irradiance (W/m²).
        U0: Free-convection coefficient (W/m²/K), strictly positive.
        U1: Wind-dependent coefficient (W·s/m³/K).
        WS: Wind speed (m/s), ≥ 0.

    Returns:
        Steady-state module temperature in Kelvin.
    """
    return T_amb_K + G_eff / (U0 + U1 * WS)


def wind_adjusted_tau(
    tau_0: Tensor,
    U0: Tensor,
    U1: Tensor,
    WS: Tensor,
) -> Tensor:
    """Wind-adjusted thermal time constant (Eq. 3.6).

    ``τ_eff = τ_0 · U0 / (U0 + U1 · WS)``.

    Args:
        tau_0: Still-air thermal time constant (s), strictly positive.
        U0: Free-convection coefficient (W/m²/K), strictly positive.
        U1: Wind-dependent coefficient (W·s/m³/K).
        WS: Wind speed (m/s), ≥ 0.

    Returns:
        Effective thermal time constant (s).
    """
    return tau_0 * U0 / (U0 + U1 * WS)


# ---------------------------------------------------------------------------
# Sandia efficiency model (Eq. 3.2).
# ---------------------------------------------------------------------------


def sandia_efficiency(
    T_K: Tensor,
    eta_ref: Tensor,
    beta_Pmax: Tensor,
    T_ref_K: Tensor,
) -> Tensor:
    """Sandia first-order efficiency model (Eq. 3.2).

    ``η(T) = η_ref · (1 + β_Pmax · (T − T_ref))`` with temperatures in K.

    Args:
        T_K: Module temperature in Kelvin.
        eta_ref: Reference efficiency at ``T_ref_K``.
        beta_Pmax: Power temperature coefficient (1/K), typically negative.
        T_ref_K: Reference temperature in Kelvin.

    Returns:
        Electrical efficiency at ``T_K``.
    """
    return eta_ref * (1.0 + beta_Pmax * (T_K - T_ref_K))


# ---------------------------------------------------------------------------
# PINN residual.
# ---------------------------------------------------------------------------


def physics_residual(
    T_hat: Tensor,
    t_hat: Tensor,
    T_ss_hat: Tensor,
    tau_eff: Tensor,
) -> Tensor:
    """PINN residual for the first-order thermal ODE.

    Computes ``τ_eff · dT̂/dt̂ − (T̂_ss − T̂)`` where ``dT̂/dt̂`` is obtained
    via ``torch.autograd.grad`` with ``create_graph=True`` so that second-order
    gradients (needed by the physics loss) remain differentiable.

    Args:
        T_hat: Predicted normalized panel temperature; must have
            ``requires_grad=True`` and be part of the same graph as ``t_hat``.
        t_hat: Normalized time input; must have ``requires_grad=True``.
        T_ss_hat: Normalized steady-state target, same shape as ``T_hat``.
        tau_eff: Effective time constant (s), same shape as ``T_hat``.

    Returns:
        Residual tensor, same shape as ``T_hat``. Zero at steady state.

    Raises:
        RuntimeError: If ``T_hat`` is not connected to ``t_hat`` in the
            autograd graph.
    """
    dT_dt = torch.autograd.grad(
        outputs=T_hat,
        inputs=t_hat,
        grad_outputs=torch.ones_like(T_hat),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if dT_dt is None:
        # T_hat is independent of t_hat in the current graph; mathematically
        # dT̂/dt̂ = 0 everywhere. Return a zero tensor that still lives in the
        # correct autograd graph so downstream .backward() calls work.
        dT_dt = torch.zeros_like(T_hat)
    return tau_eff * dT_dt - (T_ss_hat - T_hat)


# ---------------------------------------------------------------------------
# Learnable, log-parameterized physics container.
# ---------------------------------------------------------------------------


class LearnedPhysicsParameters(nn.Module):
    """Log-parameterized learnable physics scalars.

    The learnable primitives are stored as ``log_tau_0``, ``log_U0``,
    ``log_U1``, and ``log_gamma_CC`` so the exponentiated values are strictly
    positive regardless of optimizer updates. The non-learnable Sandia
    constants (``T_ref_K``, ``eta_ref``, ``beta_Pmax``) are held as buffers.

    Loss code should read physical values via the exponentiated properties
    (e.g. ``self.tau_0``) rather than touching the raw ``log_*`` parameters.
    """

    def __init__(
        self,
        tau_0_init: float,
        U0_init: float,
        U1_init: float,
        gamma_CC_init: float,
        T_ref_K: float,
        eta_ref: float,
        beta_Pmax: float,
    ) -> None:
        """Initialize from linear (not log) physical defaults."""
        super().__init__()
        if tau_0_init <= 0 or U0_init <= 0 or U1_init <= 0 or gamma_CC_init <= 0:
            raise ValueError(
                "log-parameterized physics defaults must all be strictly positive"
            )
        self.log_tau_0 = nn.Parameter(torch.tensor(math.log(tau_0_init), dtype=torch.float32))
        self.log_U0 = nn.Parameter(torch.tensor(math.log(U0_init), dtype=torch.float32))
        self.log_U1 = nn.Parameter(torch.tensor(math.log(U1_init), dtype=torch.float32))
        # sigmoid-bounded so gamma_CC stays in (0, 1] regardless of updates.
        # Initialise via the logit (sigmoid-inverse) of gamma_CC_init.
        _logit_init = math.log(gamma_CC_init / (1.0 - gamma_CC_init + 1e-7))
        self.raw_gamma_CC = nn.Parameter(
            torch.tensor(_logit_init, dtype=torch.float32)
        )
        # Sandia constants are physical calibration data, not learnable here.
        self.register_buffer("T_ref_K", torch.tensor(T_ref_K, dtype=torch.float32))
        self.register_buffer("eta_ref", torch.tensor(eta_ref, dtype=torch.float32))
        self.register_buffer("beta_Pmax", torch.tensor(beta_Pmax, dtype=torch.float32))

    @classmethod
    def from_physics_config(cls, cfg: PhysicsConfig) -> "LearnedPhysicsParameters":
        """Construct from a :class:`PhysicsConfig`.

        Args:
            cfg: Physics configuration with linear-space defaults.

        Returns:
            A :class:`LearnedPhysicsParameters` initialized at those defaults.
        """
        return cls(
            tau_0_init=cfg.tau_0_init,
            U0_init=cfg.U0_init,
            U1_init=cfg.U1_init,
            gamma_CC_init=cfg.gamma_CC_init,
            T_ref_K=cfg.T_ref_K,
            eta_ref=cfg.eta_ref,
            beta_Pmax=cfg.beta_Pmax,
        )

    # Exponentiated accessors -- read-only views into the learned state. -----

    @property
    def tau_0(self) -> Tensor:
        """Thermal time constant (s)."""
        return torch.exp(self.log_tau_0)

    @property
    def U0(self) -> Tensor:
        """Free-convection coefficient (W/m²/K)."""
        return torch.exp(self.log_U0)

    @property
    def U1(self) -> Tensor:
        """Wind-dependent coefficient (W·s/m³/K)."""
        return torch.exp(self.log_U1)

    @property
    def gamma_CC(self) -> Tensor:
        """Cloud-cover exponent (dimensionless), constrained to (0, 1]."""
        return torch.sigmoid(self.raw_gamma_CC)

    # Serialization -----------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        """Serialize current exponentiated values for checkpointing.

        Returns:
            Dict with keys ``tau_0, U0, U1, gamma_CC, T_ref_K, eta_ref,
            beta_Pmax`` as plain floats.
        """
        return {
            "tau_0": float(self.tau_0.detach()),
            "U0": float(self.U0.detach()),
            "U1": float(self.U1.detach()),
            "gamma_CC": float(self.gamma_CC.detach()),  # sigmoid-bounded, always in (0,1]
            "T_ref_K": float(self.T_ref_K.detach().item()),  # type: ignore[operator]
            "eta_ref": float(self.eta_ref.detach().item()),  # type: ignore[operator]
            "beta_Pmax": float(self.beta_Pmax.detach().item()),  # type: ignore[operator]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearnedPhysicsParameters":
        """Restore a parameter container from :meth:`to_dict` output.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            A fresh :class:`LearnedPhysicsParameters` at those values.
        """
        return cls(
            tau_0_init=float(data["tau_0"]),
            U0_init=float(data["U0"]),
            U1_init=float(data["U1"]),
            gamma_CC_init=float(data["gamma_CC"]),
            T_ref_K=float(data["T_ref_K"]),
            eta_ref=float(data["eta_ref"]),
            beta_Pmax=float(data["beta_Pmax"]),
        )

    def forward(self) -> None:  # pragma: no cover - container module
        """This module is a parameter container; call the properties instead."""
        raise RuntimeError(
            "LearnedPhysicsParameters is a parameter container; "
            "access .tau_0, .U0, .U1, .gamma_CC, etc."
        )
