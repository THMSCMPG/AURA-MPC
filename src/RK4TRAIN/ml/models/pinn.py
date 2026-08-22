"""PINN model architectures for panel temperature and efficiency prediction."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Residual block with batch norm."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        """Initialize residual block.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with skip connection.

        Args:
            x: Input tensor [B, in_dim]

        Returns:
            Output tensor [B, in_dim]
        """
        return x + self.net(x)


class PINNSurrogate(nn.Module):
    """Physics-Informed Neural Network for PV panel surrogate modeling.

    Predicts, from ONE forward pass:
    - Steady-state panel temperature (T_operating) and its uncertainty
    - 15-minute-ahead transient panel temperature (T_after_15min) and its
      uncertainty, given an independent starting temperature (T_panel_initial)
      as an input -- this is the fast NN-based lookahead the live MPC loop
      uses to evaluate candidate orientations without calling the slower
      Fortran evaluator per-candidate (see checklist D10).

    Efficiency (eta) is DELIBERATELY NOT a learned output. eta is an EXACT
    linear function of T in the underlying physics this model is trained
    against (eta = ETA_REF*(1 - BETA_T*(T - T_STC)), validated numerically
    to ~1e-11 K against the RK4TRAIN generator's own closed-form solution
    this session) -- learning it as a free 5th/6th output would let the
    network produce T/eta pairs that don't actually satisfy that relationship,
    wasting capacity re-learning something exact and risking inconsistency.
    predict_efficiency() derives eta and its uncertainty from a T prediction
    instead: since the relationship is linear, uncertainty propagates exactly
    too (eta_sigma = ETA_REF*BETA_T*T_sigma, no approximation).

    from weather conditions (T_amb, irradiance, wind), panel orientation,
    and a starting panel temperature.
    """

    # Must match RK4TRAIN's main.f90 exactly (ETA_REF, BETA_T, T_STC) -- these
    # are the physical constants the training data's closed-form solution
    # uses, not free parameters. If RK4TRAIN's constants ever change, these
    # must change with them or the analytical eta derivation becomes wrong.
    ETA_REF = 0.20
    BETA_T = 0.004
    T_STC = 298.15  # K

    def __init__(
        self,
        input_dim: int = 20,  # 7 weather + 4 panel state + 3 location + 5 time + 1 T_panel_initial
        hidden_dim: int = 128,
        num_residual_blocks: int = 4,
        num_outputs: int = 4,  # T_operating, T_operating_sigma, T_after_15min, T_after_15min_sigma
        dropout: float = 0.1,
    ) -> None:
        """Initialize PINN.

        Args:
            input_dim: Input feature dimension (weather + panel state +
                location + time + T_panel_initial). Default 20 -- see
                data.loaders.RK4TRANDataset.get_input_dim() for the
                authoritative count, don't hardcode this elsewhere.
            hidden_dim: Hidden layer dimension
            num_residual_blocks: Number of residual blocks
            num_outputs: Number of learned outputs (default 4: steady-state
                T + sigma, transient T + sigma -- NOT eta, see class docstring)
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_residual_blocks)]
        )

        # Output heads for predictions + uncertainty
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_outputs),
        )

    @classmethod
    def eta_from_T(cls, T: Tensor) -> Tensor:
        """Exact analytical efficiency from temperature -- see class docstring."""
        return cls.ETA_REF * (1.0 - cls.BETA_T * (T - cls.T_STC))

    @classmethod
    def eta_sigma_from_T_sigma(cls, T_sigma: Tensor) -> Tensor:
        """Exact uncertainty propagation through the linear eta(T) relationship."""
        return cls.ETA_REF * cls.BETA_T * T_sigma

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, input_dim] with:
                - weather: [T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure]
                - panel_state: [pv_height, pitch, roll, yaw]
                - location: [lon, lat, elevation]
                - time: [minute, hour, day_of_year, month, year]
                - T_panel_initial: starting panel temperature for the 15-min lookahead

        Returns:
            Dict with keys:
            - 'T_operating', 'T_operating_sigma': steady-state prediction
            - 'T_after_15min', 'T_after_15min_sigma': transient prediction,
              conditioned on T_panel_initial in the input
            - 'eta', 'eta_sigma': DERIVED from T_operating (not learned)
            - 'eta_after_15min', 'eta_after_15min_sigma': DERIVED from
              T_after_15min (not learned)
        """
        h = self.input_proj(x)
        h = self.residual_blocks(h)
        outputs = self.output_head(h)

        T_op = outputs[:, 0:1]
        T_op_sigma = torch.abs(outputs[:, 1:2]) + 1e-4
        T_15 = outputs[:, 2:3]
        T_15_sigma = torch.abs(outputs[:, 3:4]) + 1e-4

        return {
            "T_operating": T_op,
            "T_operating_sigma": T_op_sigma,
            "T_after_15min": T_15,
            "T_after_15min_sigma": T_15_sigma,
            "eta": self.eta_from_T(T_op),
            "eta_sigma": self.eta_sigma_from_T_sigma(T_op_sigma),
            "eta_after_15min": self.eta_from_T(T_15),
            "eta_after_15min_sigma": self.eta_sigma_from_T_sigma(T_15_sigma),
        }

    def predict(self, x: Tensor, return_uncertainty: bool = False) -> Tensor | dict[str, Tensor]:
        """Inference pass (returns only means if not requested otherwise).

        Args:
            x: Input tensor [B, input_dim]
            return_uncertainty: If True, return dict with means and sigmas

        Returns:
            Mean predictions [B, 2] for (T_operating, eta) if return_uncertainty=False,
            else full dict (see forward())
        """
        with torch.no_grad():
            outputs = self.forward(x)
            if return_uncertainty:
                return outputs
            else:
                return torch.cat(
                    [outputs["T_operating"], outputs["eta"]],
                    dim=1,
                )


class PINNEnsemble(nn.Module):
    """Ensemble of PINN models for improved uncertainty quantification."""

    def __init__(
        self,
        num_models: int = 3,
        input_dim: int = 20,
        hidden_dim: int = 128,
        num_residual_blocks: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initialize ensemble.

        Args:
            num_models: Number of models in ensemble
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            num_residual_blocks: Number of residual blocks per model
            dropout: Dropout rate
        """
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList(
            [
                PINNSurrogate(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_residual_blocks=num_residual_blocks,
                    dropout=dropout,
                )
                for _ in range(num_models)
            ]
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass through all models.

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            Dict with ensemble predictions and uncertainties
        """
        predictions = [model(x) for model in self.models]

        # Stack predictions from all models
        T_ops = torch.stack([p["T_operating"] for p in predictions], dim=0)  # [num_models, B, 1]
        etas = torch.stack([p["eta"] for p in predictions], dim=0)  # [num_models, B, 1]

        # Compute ensemble mean and std
        T_op_mean = T_ops.mean(dim=0)
        T_op_std = T_ops.std(dim=0)
        eta_mean = etas.mean(dim=0)
        eta_std = etas.std(dim=0)

        return {
            "T_operating": T_op_mean,
            "T_operating_std": T_op_std,
            "eta": eta_mean,
            "eta_std": eta_std,
        }

    def predict(self, x: Tensor) -> dict[str, Tensor]:
        """Inference pass.

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            Dict with ensemble means and stds
        """
        return self.forward(x)
