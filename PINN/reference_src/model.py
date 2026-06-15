"""PINNModel — 5-input, 2-output physics-informed neural network.

This module implements :class:`PINNModel`, a fully-connected feed-forward
network that maps five per-timestep environmental features to panel
temperature and maximum power point.

**Inputs (5 features, shape (B, 5)):**

+-----------------+---------------------------------------------+--------+
| Index           | Feature                                     | Units  |
+=================+=============================================+========+
| 0               | ``G_irradiance`` — plane-of-array irradiance| W/m²   |
+-----------------+---------------------------------------------+--------+
| 1               | ``T_amb`` — ambient temperature             | °C     |
+-----------------+---------------------------------------------+--------+
| 2               | ``WS`` — wind speed                         | m/s    |
+-----------------+---------------------------------------------+--------+
| 3               | ``azimuth`` — panel azimuth angle           | degrees|
+-----------------+---------------------------------------------+--------+
| 4               | ``height`` — panel mounting height          | m      |
+-----------------+---------------------------------------------+--------+

**Outputs (shape (B, 2)):**

* Column 0 — ``T_panel`` (°C): module temperature.
* Column 1 — ``P_mp`` (W): maximum power point power.

**Architecture:** fully-connected ``tanh`` network.

* 4–6 hidden layers, 64–128 neurons per layer (default: 4 layers × 64 neurons).
* ``tanh`` activations — smooth and infinitely differentiable, required so
  that :math:`dT/dt` can be computed via ``torch.autograd.grad``.
* Output layer has no activation (raw linear regression).
* Optional input normalization via :class:`~src.normalization.FeatureNormalizer`.

**Number of inputs:** the model accepts exactly ``N_INPUTS = 5`` features.
"""

from __future__ import annotations

from typing import cast

import torch  # noqa: F401  (used in docstring examples)
import torch.nn as nn
from torch import Tensor

from .normalization import FeatureNormalizer

__all__ = ["PINNModel", "N_INPUTS", "N_OUTPUTS"]

#: Number of input features accepted by :class:`PINNModel`.
N_INPUTS: int = 5
#: Number of output targets produced by :class:`PINNModel`.
N_OUTPUTS: int = 2


class PINNModel(nn.Module):
    """Fully-connected PINN for PV panel temperature and power.

    The network maps a batch of ``(B, 5)`` environmental feature vectors
    to ``(B, 2)`` predictions ``[T_panel, P_mp]`` at a single timestep.

    Input normalization is applied if a fitted
    :class:`~src.normalization.FeatureNormalizer` is provided at construction
    time or assigned via :attr:`normalizer`.

    Args:
        n_layers: Number of hidden fully-connected layers (default 4).
        hidden_dim: Width of each hidden layer (default 64).
        normalizer: Optional fitted :class:`~src.normalization.FeatureNormalizer`.
            If provided, inputs are standardized before being fed to the
            linear layers.
        activation: Non-linearity applied after each hidden layer.  Must be
            one of ``"tanh"`` (default, smooth everywhere — required for
            autograd-based :math:`dT/dt` computation) or ``"silu"`` (SiLU /
            Swish — often faster convergence on regression tasks).

    Raises:
        ValueError: If ``activation`` is not one of the supported strings.

    Example::

        model = PINNModel(n_layers=4, hidden_dim=64)
        x = torch.randn(32, 5)
        out = model(x)          # shape (32, 2)
        T_panel = out[:, 0]
        P_mp    = out[:, 1]

        # SiLU variant:
        model_silu = PINNModel(n_layers=5, hidden_dim=128, activation="silu")
    """

    #: Supported activation names.
    SUPPORTED_ACTIVATIONS: frozenset[str] = frozenset({"tanh", "silu"})

    def __init__(
        self,
        n_layers: int = 4,
        hidden_dim: int = 64,
        normalizer: FeatureNormalizer | None = None,
        activation: str = "tanh",
    ) -> None:
        if activation not in self.SUPPORTED_ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {sorted(self.SUPPORTED_ACTIVATIONS)}, "
                f"got {activation!r}"
            )
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.normalizer = normalizer
        self.activation = activation

        def _act() -> nn.Module:
            return nn.Tanh() if activation == "tanh" else nn.SiLU()

        # Build fully-connected trunk.
        # Layer 0: N_INPUTS → hidden_dim
        layers: list[nn.Module] = [
            nn.Linear(N_INPUTS, hidden_dim),
            _act(),
        ]
        # Layers 1..n_layers-1: hidden_dim → hidden_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), _act()]
        self.trunk = nn.Sequential(*layers)

        # Linear output layer: hidden_dim → N_OUTPUTS  (no activation)
        self.output_layer = nn.Linear(hidden_dim, N_OUTPUTS)

        # Weight initialisation: Xavier uniform + zero bias.
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for all Linear layers; zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass.

        Args:
            x: Input features, shape ``(B, 5)``.

        Returns:
            Predictions ``[T_panel, P_mp]``, shape ``(B, 2)``.
        """
        if self.normalizer is not None:
            x = self.normalizer.transform(x)
        h = self.trunk(x)                   # (B, hidden_dim)
        return cast(Tensor, self.output_layer(h))  # (B, 2)
