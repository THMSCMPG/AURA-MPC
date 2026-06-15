"""Height-dependent physics helpers for PV panel aerodynamics and irradiance.

This module implements two physical relationships that depend on the panel
mounting height above ground:

* :func:`log_law_wind_profile` — effective heat-transfer coefficient
  :math:`U_L(z)` via the log-law wind profile.
* :func:`bifacial_view_factor` — ground-albedo view factor
  :math:`F_\\mathrm{ground}(z)` for bifacial panels.

All functions accept and return :class:`torch.Tensor` objects and are
differentiable end-to-end through :mod:`torch.autograd`.

**Log-law wind profile (Fuentes model):**

.. math::

    U_L(z) = h_\\mathrm{conv} \\cdot
        \\left(WS \\cdot
            \\frac{\\ln(z/z_0)}{\\ln(z_\\mathrm{ref}/z_0)}
        \\right)^{0.8}

where :math:`z_0 = 0.03\\,\\text{m}` (urban/suburban roughness length) and
:math:`z_\\mathrm{ref} = 10\\,\\text{m}` (standard meteorological reference
height).

**Bifacial view factor:**

.. math::

    F_\\mathrm{ground}(z) =
        0.5 \\cdot \\left(1 - \\cos\\!\\left(
            \\arctan\\!\\left(\\frac{z}{L_\\mathrm{panel}/2}\\right)
        \\right)\\right)

where :math:`L_\\mathrm{panel} = 1.65\\,\\text{m}` (default panel length).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["log_law_wind_profile", "bifacial_view_factor"]

# ---------------------------------------------------------------------------
# Default physical constants.
# ---------------------------------------------------------------------------

#: Aerodynamic roughness length for open-country / suburban terrain [m].
_Z0: float = 0.03
#: Standard meteorological mast reference height [m].
_Z_REF: float = 10.0
#: Default empirical convection base coefficient [W/(m²·K·(m/s)^0.8)].
_H_CONV_BASE: float = 10.0
#: Wind-profile exponent (Fuentes model).
_WIND_EXP: float = 0.8
#: Default PV panel length for view-factor geometry [m].
_L_PANEL: float = 1.65


# ---------------------------------------------------------------------------
# Public functions.
# ---------------------------------------------------------------------------


def log_law_wind_profile(
    WS: Tensor,
    height: Tensor,
    *,
    z0: float = _Z0,
    z_ref: float = _Z_REF,
    h_conv_base: float = _H_CONV_BASE,
) -> Tensor:
    """Compute the overall heat-transfer coefficient via the log-law profile.

    The effective wind speed at height *z* is derived from the reference
    measurement at *z_ref* via the log-law wind profile. The heat-transfer
    coefficient then scales as the 0.8 power of the effective wind speed:

    .. math::

        U_L(z) = h_\\mathrm{conv} \\cdot
            \\left(WS \\cdot
                \\frac{\\ln(z/z_0)}{\\ln(z_\\mathrm{ref}/z_0)}
            \\right)^{0.8}

    Heights below *z0* are clamped to *z0* so the argument of the logarithm
    is always ≥ 1, keeping :math:`U_L \\ge 0`.

    Args:
        WS: Wind speed at reference height *z_ref* [m/s], shape ``(B,)``.
        height: Panel mounting height above ground [m], shape ``(B,)``.
        z0: Roughness length [m] (default 0.03 m).
        z_ref: Reference anemometer height [m] (default 10 m).
        h_conv_base: Empirical convection coefficient
            [W/(m²·K·(m/s)^0.8)] (default 10.0).

    Returns:
        Overall heat-transfer coefficient :math:`U_L` [W/(m²·K)], same
        shape as *WS*.
    """
    log_ref = math.log(z_ref / z0)                        # scalar constant
    log_z = torch.log(height.clamp(min=z0) / z0)          # (B,)
    ws_eff = WS * log_z / log_ref                          # (B,)
    return h_conv_base * ws_eff.clamp(min=0.0) ** _WIND_EXP  # (B,)


def bifacial_view_factor(
    height: Tensor,
    *,
    L_panel: float = _L_PANEL,
) -> Tensor:
    """Ground-albedo view factor for a bifacial panel at height *z*.

    Models the fraction of the lower hemisphere that has a direct line-of-
    sight to the ground as a function of mounting height and panel length:

    .. math::

        F_\\mathrm{ground}(z) =
            0.5 \\cdot \\left(1 -
                \\cos\\!\\left(\\arctan\\!\\left(
                    \\frac{z}{L_\\mathrm{panel}/2}
                \\right)\\right)
            \\right)

    Boundary behaviour:

    * At :math:`z = 0`: :math:`F_\\mathrm{ground} = 0` (panel on ground,
      no view of ground below).
    * As :math:`z \\to \\infty`: :math:`F_\\mathrm{ground} \\to 0.5` (full
      lower hemisphere viewed).

    Args:
        height: Panel mounting height above ground [m], shape ``(B,)`` or
            scalar.  Must be ≥ 0; negative values are clamped to 0.
        L_panel: Panel length [m] (default 1.65 m).

    Returns:
        View factor [dimensionless, 0–0.5], same shape as *height*.
    """
    half_L = L_panel / 2.0
    angle = torch.atan(height.clamp(min=0.0) / half_L)    # (B,)
    return 0.5 * (1.0 - torch.cos(angle))                  # (B,)
