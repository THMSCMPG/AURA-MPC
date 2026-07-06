"""PINN total loss: data fidelity + physics residual.

:class:`TotalLoss` composes the data-fidelity MSE and the two physics loss
components (:func:`~src.physics_loss.compute_ode_residual` and
:func:`~src.physics_loss.compute_bc_loss`) into a single weighted scalar:

.. math::

    L = w_\\mathrm{data} \\cdot \\mathrm{MSE}_\\mathrm{data}
      + w_\\mathrm{phys} \\cdot \\mathrm{MSE}_\\mathrm{phys}

where

.. math::

    \\mathrm{MSE}_\\mathrm{data}
        &= \\mathrm{MSE}(T_\\mathrm{pred}, T_\\mathrm{target})
         + \\mathrm{MSE}(P_\\mathrm{pred}, P_\\mathrm{target}), \\\\
    \\mathrm{MSE}_\\mathrm{phys}
        &= \\mathrm{mean}(r_\\mathrm{ODE}^2)
         + \\mathrm{mean}(r_\\mathrm{BC}^2).

Default weights: :math:`w_\\mathrm{data} = 1.0`,
:math:`w_\\mathrm{phys} = 0.1`.
"""

from __future__ import annotations

from typing import Any

import torch.nn.functional as F
from torch import Tensor

from .physics_loss import compute_bc_loss, compute_ode_residual

__all__ = ["TotalLoss"]


class TotalLoss:
    """Weighted sum of data MSE and physics residual losses.

    Args:
        w_data: Weight for the data-fidelity MSE terms (default 1.0).
        w_phys: Weight for the physics residual terms (default 0.1).
        params: Optional dict of physical constants forwarded to
            :func:`~src.physics_loss.compute_ode_residual`.  See that
            function's docstring for supported keys.

    Example::

        loss_fn = TotalLoss(w_data=1.0, w_phys=0.1)
        losses = loss_fn(
            T_pred, P_pred, T_target, P_target,
            T_amb, dT_dt, G, WS, height,
        )
        losses["total"].backward()
    """

    def __init__(
        self,
        w_data: float = 1.0,
        w_phys: float = 0.1,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.w_data = w_data
        self.w_phys = w_phys
        self.params = params

    def __call__(
        self,
        T_pred: Tensor,
        P_pred: Tensor,
        T_target: Tensor,
        P_target: Tensor,
        T_amb: Tensor,
        dT_dt: Tensor,
        G: Tensor,
        WS: Tensor,
        height: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the total PINN loss.

        Args:
            T_pred: Predicted panel temperature [°C], shape ``(B,)``.
            P_pred: Predicted maximum power point [W], shape ``(B,)``.
            T_target: Ground-truth panel temperature [°C], shape ``(B,)``.
            P_target: Ground-truth maximum power point [W], shape ``(B,)``.
            T_amb: Ambient temperature [°C], shape ``(B,)``.
            dT_dt: Temperature time derivative [°C/s], shape ``(B,)``.
            G: Plane-of-array irradiance [W/m²], shape ``(B,)``.
            WS: Wind speed at reference height [m/s], shape ``(B,)``.
            height: Panel mounting height above ground [m], shape ``(B,)``.

        Returns:
            Dict with keys:

            * ``'total'`` — weighted total loss (scalar).
            * ``'mse_data_T'`` — MSE on temperature (scalar).
            * ``'mse_data_P'`` — MSE on power (scalar).
            * ``'mse_phys'`` — mean-squared ODE residual (scalar).
            * ``'mse_bc'`` — mean-squared boundary constraint (scalar).
        """
        mse_T = F.mse_loss(T_pred, T_target)
        mse_P = F.mse_loss(P_pred, P_target)
        mse_data = mse_T + mse_P

        ode_res = compute_ode_residual(
            T_pred, dT_dt, G, T_amb, WS, height, params=self.params
        )
        bc_res = compute_bc_loss(T_pred, T_amb)
        mse_phys = (ode_res ** 2).mean() + bc_res.mean()

        total = self.w_data * mse_data + self.w_phys * mse_phys

        return {
            "total": total,
            "mse_data_T": mse_T,
            "mse_data_P": mse_P,
            "mse_phys": (ode_res ** 2).mean(),
            "mse_bc": bc_res.mean(),
        }
