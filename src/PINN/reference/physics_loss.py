"""Physics loss components for the Fuentes thermal PINN.

This module provides the per-sample physics residual functions used to
build the physics-informed loss during training.  All functions operate on
:class:`torch.Tensor` objects and are end-to-end differentiable.

**Functions exported:**

* :func:`compute_uL` — overall heat-transfer coefficient via the
  Fuentes/log-law model.
* :func:`compute_ode_residual` — Fuentes ODE residual at a single
  timestep, one value per batch sample.
* :func:`compute_bc_loss` — boundary-constraint loss penalising
  sub-ambient panel temperatures.

**Fuentes ODE (simplified form used here):**

.. math::

    C_\\mathrm{panel} \\frac{dT_\\mathrm{panel}}{dt}
    = \\alpha G
      - U_L(WS, z) \\cdot (T_\\mathrm{panel} - T_\\mathrm{amb})
      - \\eta_\\mathrm{cell} \\cdot G

where :math:`U_L` is computed via the log-law wind profile in
:mod:`src.height_physics`.

**Boundary constraint:**

.. math::

    r_\\mathrm{BC}(t) = \\max(0,\\, T_\\mathrm{amb}(t) - T_\\mathrm{panel}(t))^2

This penalises predictions where the panel is colder than ambient, which
is physically impossible under irradiance.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .height_physics import log_law_wind_profile

__all__ = ["compute_uL", "compute_ode_residual", "compute_bc_loss"]

# ---------------------------------------------------------------------------
# Default physical parameters.
# ---------------------------------------------------------------------------

#: Panel thermal capacitance [J/K].
_C_PANEL: float = 6_000.0
#: Solar absorptance [-].
_ALPHA: float = 0.9
#: Cell electrical efficiency [-].
_ETA_CELL: float = 0.18


# ---------------------------------------------------------------------------
# Public functions.
# ---------------------------------------------------------------------------


def compute_uL(
    WS: Tensor,
    height: Tensor,
    *,
    params: dict[str, Any] | None = None,
) -> Tensor:
    """Compute the overall heat-transfer coefficient :math:`U_L`.

    Delegates to :func:`src.height_physics.log_law_wind_profile` using
    the Fuentes model parameters.

    Args:
        WS: Wind speed at reference height [m/s], shape ``(B,)``.
        height: Panel mounting height above ground [m], shape ``(B,)``.
        params: Optional dict to override default constants.  Supported
            keys: ``'z0'``, ``'z_ref'``, ``'h_conv_base'``.

    Returns:
        Heat-transfer coefficient :math:`U_L` [W/(m²·K)], shape ``(B,)``.
    """
    p = params or {}
    return log_law_wind_profile(
        WS,
        height,
        z0=float(p.get("z0", 0.03)),
        z_ref=float(p.get("z_ref", 10.0)),
        h_conv_base=float(p.get("h_conv_base", 10.0)),
    )


def compute_ode_residual(
    T_panel: Tensor,
    dT_dt: Tensor,
    G: Tensor,
    T_amb: Tensor,
    WS: Tensor,
    height: Tensor,
    *,
    params: dict[str, Any] | None = None,
) -> Tensor:
    """Fuentes ODE residual at a single timestep.

    Computes:

    .. math::

        r_\\mathrm{ODE} =
            C_\\mathrm{panel} \\cdot \\frac{dT_\\mathrm{panel}}{dt}
            - \\bigl[
                \\alpha G
                - U_L(WS, z) \\cdot (T_\\mathrm{panel} - T_\\mathrm{amb})
                - \\eta_\\mathrm{cell} \\cdot G
            \\bigr]

    The residual is zero at the exact physical solution.  When
    *T_panel* or *dT_dt* carry gradient information (e.g. when they are
    outputs of a neural network or are computed via
    :func:`torch.autograd.grad`), the returned tensor retains its
    ``grad_fn`` and can be backpropagated through.

    Args:
        T_panel: Predicted panel temperature [°C], shape ``(B,)``.
        dT_dt: Time derivative :math:`dT_\\mathrm{panel}/dt` [°C/s],
            shape ``(B,)``.  Can be computed externally via autograd or
            finite differences.
        G: Plane-of-array irradiance [W/m²], shape ``(B,)``.
        T_amb: Ambient temperature [°C], shape ``(B,)``.
        WS: Wind speed at reference height [m/s], shape ``(B,)``.
        height: Panel mounting height above ground [m], shape ``(B,)``.
        params: Optional dict to override default physical constants.
            Supported keys: ``'C_panel'``, ``'alpha'``, ``'eta_cell'``,
            ``'z0'``, ``'z_ref'``, ``'h_conv_base'``.

    Returns:
        ODE residual :math:`r_\\mathrm{ODE}` [W/K], shape ``(B,)``.
        Inherits ``grad_fn`` from inputs.
    """
    p = params or {}
    C_panel = float(p.get("C_panel", _C_PANEL))
    alpha = float(p.get("alpha", _ALPHA))
    eta_cell = float(p.get("eta_cell", _ETA_CELL))

    U_L = compute_uL(WS, height, params=p)                 # (B,)

    rhs = alpha * G - U_L * (T_panel - T_amb) - eta_cell * G
    return C_panel * dT_dt - rhs                            # (B,)


def compute_bc_loss(T_panel: Tensor, T_amb: Tensor) -> Tensor:
    """Boundary-constraint loss penalising sub-ambient panel temperature.

    Returns the element-wise penalty:

    .. math::

        r_\\mathrm{BC}(t) =
            \\max(0,\\, T_\\mathrm{amb}(t) - T_\\mathrm{panel}(t))^2

    A panel temperature below ambient is physically impossible under any
    positive irradiance.  This term enforces the soft constraint
    :math:`T_\\mathrm{panel} \\ge T_\\mathrm{amb}` during training.

    Args:
        T_panel: Predicted panel temperature [°C], shape ``(B,)``.
        T_amb: Ambient temperature [°C], shape ``(B,)``.

    Returns:
        Per-sample constraint violation squared [K²], shape ``(B,)``.
        Zero whenever :math:`T_\\mathrm{panel} \\ge T_\\mathrm{amb}`.
    """
    return torch.relu(T_amb - T_panel) ** 2                # (B,)
