"""Fuentes ODE physics loss for the PINNSurrogate.

This module contains:

* :func:`fuentes_ode_residual` — residual of the Fuentes energy-balance ODE.
* :func:`boundary_violation` — sub-ambient temperature penalty.
* :func:`pinn_loss` — combined data + physics loss used during training.

All functions are pure: they accept and return ``torch.Tensor`` objects and
are compatible with ``torch.autograd`` end-to-end.

**Fuentes ODE (energy balance):**

.. math::

    C_m \\frac{dT}{dt} = \\alpha G
        - h(WS_{\\mathrm{eff}})(T - T_{\\mathrm{amb}})
        - \\varepsilon\\sigma(T_K^4 - T_{\\mathrm{sky},K}^4)
        - \\eta(T) G

where:

* :math:`h(WS) = 5.7 + 3.8 \\cdot WS_{\\mathrm{eff}}` (McAdams convection)
* :math:`\\eta(T) = \\eta_{\\mathrm{ref}}(1 + \\beta(T - T_{\\mathrm{ref}}))`
* :math:`T_{\\mathrm{sky}} = T_{\\mathrm{amb}} - 20` [°C]
* :math:`C_m = 11\\,000` J/(m²·K)
* :math:`\\alpha = 0.9`, :math:`\\varepsilon = 0.85`,
  :math:`\\sigma = 5.67 \\times 10^{-8}` W/(m²·K⁴)
* :math:`\\eta_{\\mathrm{ref}} = 0.20`, :math:`\\beta = -0.004` K⁻¹,
  :math:`T_{\\mathrm{ref}} = 25` °C
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["fuentes_ode_residual", "boundary_violation", "pinn_loss"]

# ---------------------------------------------------------------------------
# Default physical constants.
# ---------------------------------------------------------------------------

_C_M: float = 11_000.0       # thermal mass per unit area [J/(m² K)]
_ALPHA: float = 0.9           # solar absorptivity [-]
_EPS: float = 0.85            # thermal emissivity [-]
_SIGMA: float = 5.67e-8       # Stefan-Boltzmann [W/(m² K⁴)]
_ETA_REF: float = 0.20        # reference PV efficiency [-]
_BETA: float = -0.004         # temperature derating coefficient [1/K]
_T_REF: float = 25.0          # reference temperature [°C]
_T_SKY_OFFSET: float = 20.0   # sky-temperature depression [K]
_KELVIN: float = 273.15       # °C → K offset

#: Default timestep assumed when ``t`` is absent or has < 2 points [s].
#: Matches 15-minute resolution over a 24-hour horizon with 96 timesteps.
_DEFAULT_TIMESTEP_SECONDS: float = 900.0


# ---------------------------------------------------------------------------
# Public functions.
# ---------------------------------------------------------------------------


def fuentes_ode_residual(
    T_panel: Tensor,
    t: Tensor,
    G: Tensor,
    WS_eff: Tensor,
    T_amb: Tensor,
    params: dict[str, Any] | None = None,
) -> Tensor:
    """Compute the Fuentes ODE residual at each timestep.

    ODE: ``C_m · dT/dt = α·G − h(WS)·(T − T_amb) − ε·σ·(T_K⁴ − T_sky_K⁴) − η(T)·G``

    ``dT/dt`` is computed via :func:`torch.gradient` (central differences)
    on the ``T_panel`` tensor, so it is differentiable through the autograd
    graph.

    Args:
        T_panel: Predicted panel temperature [°C], shape ``(B, N)``.
        t: Time vector [s], shape ``(N,)``.  Used to determine the timestep
            spacing for the gradient.  If ``None`` or fewer than two points,
            a default 900-second spacing is assumed.
        G: Plane-of-array irradiance [W/m²], shape ``(B,)`` or ``(B, N)``.
        WS_eff: Effective wind speed at panel height [m/s], shape ``(B,)``
            or ``(B, N)``.
        T_amb: Ambient temperature [°C], shape ``(B,)`` or ``(B, N)``.
        params: Optional dict to override default physical constants.  Supported
            keys: ``C_m``, ``alpha``, ``eps``, ``sigma``, ``eta_ref``,
            ``beta``, ``T_ref``, ``T_sky_offset``.

    Returns:
        Residual tensor of shape ``(B, N)``.  Zero everywhere at the exact
        physical solution.
    """
    p = params or {}
    C_m = float(p.get("C_m", _C_M))
    alpha = float(p.get("alpha", _ALPHA))
    eps = float(p.get("eps", _EPS))
    sigma = float(p.get("sigma", _SIGMA))
    eta_ref = float(p.get("eta_ref", _ETA_REF))
    beta = float(p.get("beta", _BETA))
    T_ref = float(p.get("T_ref", _T_REF))
    T_sky_offset = float(p.get("T_sky_offset", _T_SKY_OFFSET))

    # --- time-derivative via central differences -------------------------
    n_t = T_panel.shape[1]
    if t is not None and t.numel() >= 2:
        dt = float((t[-1] - t[0]) / max(n_t - 1, 1))
    else:
        dt = _DEFAULT_TIMESTEP_SECONDS

    # torch.gradient returns a list; we want the gradient along dim=1.
    dT_dt = torch.gradient(T_panel, spacing=(dt,), dim=1)[0]   # (B, N)

    # --- broadcast scalar inputs to (B, N) if needed --------------------
    def _bcast(x: Tensor) -> Tensor:
        if x.dim() == 1:
            return x.unsqueeze(-1).expand_as(T_panel)
        return x

    G_b = _bcast(G)
    WS_b = _bcast(WS_eff)
    T_amb_b = _bcast(T_amb)

    # --- physics --------------------------------------------------------
    h_ws = 5.7 + 3.8 * WS_b                                      # McAdams
    eta_T = eta_ref * (1.0 + beta * (T_panel - T_ref))            # efficiency
    T_sky = T_amb_b - T_sky_offset                                 # sky temp [°C]

    # Radiation in Kelvin.
    T_K = T_panel + _KELVIN
    T_sky_K = T_sky + _KELVIN

    rhs = (
        alpha * G_b
        - h_ws * (T_panel - T_amb_b)
        - eps * sigma * (T_K ** 4 - T_sky_K ** 4)
        - eta_T * G_b
    )

    return C_m * dT_dt - rhs                                       # (B, N)


def boundary_violation(T_panel: Tensor, T_amb: Tensor) -> Tensor:
    """Sub-ambient temperature penalty.

    Returns the mean of ``max(0, T_amb − T_panel)`` over all batch items and
    timesteps.  Penalises predictions where the panel is colder than ambient.

    Args:
        T_panel: Predicted panel temperature [°C], shape ``(B, N)``.
        T_amb: Ambient temperature [°C], shape ``(B,)`` or ``(B, N)``.

    Returns:
        Scalar tensor; zero when all ``T_panel ≥ T_amb``.
    """
    if T_amb.dim() == 1:
        T_amb_b = T_amb.unsqueeze(-1).expand_as(T_panel)
    else:
        T_amb_b = T_amb
    return F.relu(T_amb_b - T_panel).mean()


def pinn_loss(
    T_pred: Tensor,
    P_pred: Tensor,
    T_target: Tensor,
    P_target: Tensor,
    T_amb: Tensor,
    t: Tensor,
    G: Tensor,
    WS_eff: Tensor,
    w_data: float = 1.0,
    w_phys: float = 0.1,
) -> dict[str, Tensor]:
    """Compute the total PINN loss.

    .. math::

        L = w_{\\mathrm{data}} \\bigl[
                \\mathrm{MSE}(T_{\\mathrm{pred}}, T_{\\mathrm{target}})
              + \\mathrm{MSE}(P_{\\mathrm{pred}}, P_{\\mathrm{target}})
            \\bigr]
          + w_{\\mathrm{phys}} \\cdot \\overline{r^2}
          + w_{\\mathrm{data}} \\cdot \\overline{\\max(0, T_{\\mathrm{amb}} - T_{\\mathrm{pred}})}

    where :math:`r` is the Fuentes ODE residual.

    Args:
        T_pred: Predicted panel temperature [°C], shape ``(B, N)``.
        P_pred: Predicted maximum power point [W], shape ``(B, N)``.
        T_target: Ground-truth panel temperature [°C], shape ``(B, N)``.
        P_target: Ground-truth maximum power point [W], shape ``(B, N)``.
        T_amb: Ambient temperature [°C], shape ``(B,)`` or ``(B, N)``.
        t: Time vector [s], shape ``(N,)``.
        G: Plane-of-array irradiance [W/m²], shape ``(B,)`` or ``(B, N)``.
        WS_eff: Effective wind speed [m/s], shape ``(B,)`` or ``(B, N)``.
        w_data: Weight for the data-fidelity MSE terms (default ``1.0``).
        w_phys: Weight for the physics residual term (default ``0.1``).

    Returns:
        Dict with keys ``'total'``, ``'mse_data_T'``, ``'mse_data_P'``,
        ``'mse_phys'``, and ``'boundary'``.  All values are scalar tensors.
    """
    mse_T = F.mse_loss(T_pred, T_target)
    mse_P = F.mse_loss(P_pred, P_target)

    residual = fuentes_ode_residual(T_pred, t, G, WS_eff, T_amb)
    mse_phys = (residual ** 2).mean()

    bv = boundary_violation(T_pred, T_amb)

    total = w_data * (mse_T + mse_P) + w_phys * mse_phys + w_data * bv

    return {
        "total": total,
        "mse_data_T": mse_T,
        "mse_data_P": mse_P,
        "mse_phys": mse_phys,
        "boundary": bv,
    }
