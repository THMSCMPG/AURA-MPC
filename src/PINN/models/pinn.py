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

    Predicts:
    - Panel temperature (T_operating)
    - Electrical efficiency (η)

    from weather conditions (T_amb, irradiance, wind) and panel orientation.
    """

    def __init__(
        self,
        input_dim: int = 18,  # 7 weather + 4 panel state + 3 location + 4 time
        hidden_dim: int = 128,
        num_residual_blocks: int = 4,
        num_outputs: int = 4,  # T_operating, T_operating_sigma, eta, eta_sigma
        dropout: float = 0.1,
    ) -> None:
        """Initialize PINN.

        Args:
            input_dim: Input feature dimension (weather + panel state + location + time)
            hidden_dim: Hidden layer dimension
            num_residual_blocks: Number of residual blocks
            num_outputs: Number of outputs (default 4 for mean+sigma)
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

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, input_dim] with:
                - weather: [T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure]
                - panel_state: [pv_height, pitch, roll, yaw]
                - location: [lat, lon, elevation]
                - time: [hour, day, month, year, ...]

        Returns:
            Dict with keys:
            - 'T_operating': Panel temperature [B, 1]
            - 'T_operating_sigma': Temperature uncertainty [B, 1]
            - 'eta': Efficiency [B, 1]
            - 'eta_sigma': Efficiency uncertainty [B, 1]
        """
        h = self.input_proj(x)
        h = self.residual_blocks(h)
        outputs = self.output_head(h)

        # Split outputs
        T_op = outputs[:, 0:1]
        T_sigma = torch.abs(outputs[:, 1:2]) + 1e-4  # Ensure positive
        eta = outputs[:, 2:3]
        eta_sigma = torch.abs(outputs[:, 3:4]) + 1e-4  # Ensure positive

        return {
            "T_operating": T_op,
            "T_operating_sigma": T_sigma,
            "eta": eta,
            "eta_sigma": eta_sigma,
        }

    def predict(self, x: Tensor, return_uncertainty: bool = False) -> Tensor | dict[str, Tensor]:
        """Inference pass (returns only means if not requested otherwise).

        Args:
            x: Input tensor [B, input_dim]
            return_uncertainty: If True, return dict with means and sigmas

        Returns:
            Mean predictions [B, 2] for (T_operating, eta) if return_uncertainty=False,
            else dict with means and sigmas
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
        input_dim: int = 18,
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
