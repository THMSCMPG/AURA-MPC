"""Dual-head multimodal PINN architecture (design doc §6.1).

Implements the sky-image encoder, the residual-block trunk, and the
``DualHeadPINN`` module with a temperature regression head and a 5-way
routing-classification head. The network holds a
:class:`LearnedPhysicsParameters` submodule so its ``log_*`` scalars
participate in optimizer updates.

Also implements :class:`PINNSurrogate`, a purely-passive feed-forward
surrogate for the Fuentes ODE that maps 6 scalar environmental inputs to
panel-temperature and power time-series over a 24-hour horizon.

Public interface: :class:`SkyImageEncoder`, :class:`ResidualBlock`,
:class:`DualHeadPINN`, :class:`PINNSurrogate`,
:func:`log_law_ws_eff`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .physics import LearnedPhysicsParameters

__all__ = ["SkyImageEncoder", "ResidualBlock", "DualHeadPINN", "PINNSurrogate", "log_law_ws_eff"]


# ---------------------------------------------------------------------------
# Log-law wind-speed pre-processing (exposed for use in training loops).
# ---------------------------------------------------------------------------

#: Smooth-terrain aerodynamic roughness length [m].
_Z0: float = 0.01
#: Standard meteorological mast height [m].
_Z_REF: float = 10.0


def log_law_ws_eff(WS: Tensor, height: Tensor) -> Tensor:
    """Apply log-law wind profile to convert hub-height wind to panel height.

    ``WS_eff = WS * log(height / z0) / log(z_ref / z0)``

    where ``z0 = 0.01 m`` (smooth-terrain roughness) and ``z_ref = 10.0 m``
    (standard met-mast reference height). Heights below ``z0`` are clamped to
    ``z0`` so the result is always non-negative.

    Args:
        WS: Wind speed at reference height [m/s], shape ``(B,)``.
        height: Panel mounting height [m], shape ``(B,)``.

    Returns:
        Effective wind speed at panel height [m/s], shape ``(B,)``,
        clamped to ``>= 0``.
    """
    log_denom = math.log(_Z_REF / _Z0)  # scalar, constant
    log_numer = torch.log(height.clamp(min=_Z0) / _Z0)
    return (WS * log_numer / log_denom).clamp(min=0.0)


class SkyImageEncoder(nn.Module):
    """Small CNN encoder for 32×32 RGB sky images (design doc §6.1).

    Three stride-2 conv stages (3→16→32→32), each with BatchNorm2d + ReLU,
    followed by global average pooling to a fixed ``image_embed_dim``-long
    embedding.
    """

    def __init__(self, image_embed_dim: int = 32) -> None:
        """Create the encoder.

        Args:
            image_embed_dim: Must equal 32; the conv stack's final channel
                count is fixed and the global pool produces exactly 32
                features.

        Raises:
            ValueError: If ``image_embed_dim`` is not 32.
        """
        super().__init__()
        if image_embed_dim != 32:
            raise ValueError(
                f"SkyImageEncoder requires image_embed_dim=32, got {image_embed_dim}"
            )
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.image_embed_dim = image_embed_dim

    def forward(self, image: Tensor) -> Tensor:
        """Encode an image batch.

        Args:
            image: ``(B, 3, H, W)`` float tensor.

        Returns:
            ``(B, image_embed_dim)`` embedding tensor.
        """
        x = self.relu(self.bn1(self.conv1(image)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        return torch.flatten(x, start_dim=1)


class ResidualBlock(nn.Module):
    """Tanh residual block for the shared trunk (design doc §6.1).

    ``forward(x) = tanh(x + fc2(tanh(fc1(x))))``. No dropout, no BatchNorm —
    PINN training with autograd-differentiated residuals is sensitive to
    both.
    """

    def __init__(self, hidden_dim: int) -> None:
        """Create a residual block of width ``hidden_dim``."""
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the residual block.

        Args:
            x: ``(B, hidden_dim)`` input features.

        Returns:
            ``(B, hidden_dim)`` output features, same dtype/device.
        """
        h = torch.tanh(self.fc1(x))
        return torch.tanh(x + self.fc2(h))


class DualHeadPINN(nn.Module):
    """Shared-trunk PINN with temperature and simulation-routing heads.

    The model fuses 7 numeric features with a 32-dim sky-image embedding,
    runs the result through a ``hidden_dim``-wide residual-block trunk, and
    branches into a scalar temperature head and a ``num_routes``-way
    routing-classifier head.

    The learnable physics scalars are held as a submodule so their
    ``log_*`` parameters receive gradient updates through
    ``model.parameters()``.
    """

    def __init__(self, model_cfg: ModelConfig, physics: LearnedPhysicsParameters) -> None:
        """Build the dual-head architecture from a :class:`ModelConfig`.

        Args:
            model_cfg: Architecture sizes. Field values are read; the
                dataclass itself is not retained.
            physics: Learnable physics container. Stored as a submodule so
                its parameters are visible to the optimizer.
        """
        super().__init__()
        self.num_numeric_features = model_cfg.num_numeric_features
        self.image_embed_dim = model_cfg.image_embed_dim
        self.hidden_dim = model_cfg.hidden_dim
        self.num_residual_blocks = model_cfg.num_residual_blocks
        self.route_hidden = model_cfg.route_hidden
        self.num_routes = model_cfg.num_routes

        self.image_encoder = SkyImageEncoder(image_embed_dim=self.image_embed_dim)
        self.input_proj = nn.Linear(
            self.num_numeric_features + self.image_embed_dim, self.hidden_dim
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(self.hidden_dim) for _ in range(self.num_residual_blocks)]
        )
        self.temp_head = nn.Linear(self.hidden_dim, 1)
        self.route_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.route_hidden),
            nn.Tanh(),
            nn.Linear(self.route_hidden, self.num_routes),
        )
        # Pose head: predicts (pitch, yaw, roll, z) from the shared trunk.
        # Outputs raw values; activations are applied in forward() so the
        # physical range constraints are structural, not loss-based.
        pose_hidden = getattr(model_cfg, 'pose_hidden', 64)
        self.pose_head = nn.Sequential(
            nn.Linear(self.hidden_dim, pose_hidden),
            nn.Tanh(),
            nn.Linear(pose_hidden, 4),
        )
        # Submodule so log_* parameters appear in .parameters().
        self.physics = physics

    def _features(self, numeric: Tensor, image: Tensor) -> Tensor:
        """Compute the trunk output shared by both heads."""
        image_embed = self.image_encoder(image)
        fused = torch.cat([numeric, image_embed], dim=-1)
        projected = torch.tanh(self.input_proj(fused))
        return self.trunk(projected)  # type: ignore[no-any-return]

    def forward(self, numeric: Tensor, image: Tensor) -> dict[str, Tensor]:
        """Run a forward pass.

        Args:
            numeric: ``(B, num_numeric_features)`` numeric inputs.
            image: ``(B, 3, H, W)`` sky-image inputs.

        Returns:
            Dict with keys:
              ``T_hat``: ``(B, 1)`` normalized panel temperature.
              ``route_logits``: ``(B, num_routes)`` pre-softmax logits.
              ``features``: ``(B, hidden_dim)`` trunk embedding.
        """
        features = self._features(numeric, image)
        raw_pose = self.pose_head(features)  # (B, 4) raw
        # Structural range constraints (same bounds as ControlConfig defaults):
        #   pitch : tanh * 35 deg
        #   yaw   : tanh * 180 deg
        #   roll  : tanh * 25 deg
        #   z     : sigmoid * 3 m  (always non-negative)
        pose_pitch = torch.tanh(raw_pose[:, 0:1]) * 35.0
        pose_yaw   = torch.tanh(raw_pose[:, 1:2]) * 180.0
        pose_roll  = torch.tanh(raw_pose[:, 2:3]) * 25.0
        pose_z     = torch.sigmoid(raw_pose[:, 3:4]) * 3.0
        pose = torch.cat([pose_pitch, pose_yaw, pose_roll, pose_z], dim=-1)  # (B, 4)
        return {
            "T_hat": self.temp_head(features),
            "route_logits": self.route_head(features),
            "pose": pose,  # (B, 4) [pitch_deg, yaw_deg, roll_deg, z_m]
            "features": features,
        }

    def predict_with_uncertainty(
        self, numeric: Tensor, image: Tensor
    ) -> dict[str, Tensor]:
        """Forward pass with softmax probabilities and routing uncertainty.

        Args:
            numeric: ``(B, num_numeric_features)`` numeric inputs.
            image: ``(B, 3, H, W)`` sky-image inputs.

        Returns:
            Dict with keys ``T_hat``, ``route_probs`` (softmax over logits),
            and ``route_uncertainty`` = ``1 − max(route_probs)``. For a
            perfectly uniform distribution over ``K`` routes, the uncertainty
            equals ``1 − 1/K``.
        """
        out = self.forward(numeric, image)
        probs = torch.softmax(out["route_logits"], dim=-1)
        uncertainty = 1.0 - probs.max(dim=-1).values
        return {
            "T_hat": out["T_hat"],
            "route_probs": probs,
            "route_uncertainty": uncertainty,
            "pose": out["pose"],  # (B, 4) [pitch_deg, yaw_deg, roll_deg, z_m]
        }


# ---------------------------------------------------------------------------
# PINNSurrogate — purely passive 6-input → 2 time-series surrogate.
# ---------------------------------------------------------------------------

#: Number of input features fed to the linear layers (6 base + VF).
_SURROGATE_N_INPUTS: int = 7


class PINNSurrogate(nn.Module):
    """Physics-Informed Neural Network surrogate for PV panel temperature and power.

    This is a *purely passive* surrogate: it does not route solvers or make
    fidelity decisions. Its sole purpose is to be a fast, differentiable
    approximation of the Fuentes ODE.

    **Inputs (6 scalars or batch):**

    * ``G`` — plane-of-array irradiance [W/m²]
    * ``T_amb`` — ambient temperature [°C]
    * ``WS`` — wind speed at reference height [m/s]
    * ``tilt`` — panel tilt angle [degrees, 0–90]
    * ``azimuth`` — panel azimuth angle [degrees, −180–180]
    * ``height`` — panel mounting height above ground [m]

    **Outputs (time-series tensors over 24-hour horizon):**

    * ``T_panel`` — panel temperature [°C], shape ``(B, n_timesteps)``
    * ``P_mp`` — maximum power point [W], shape ``(B, n_timesteps)``

    **Pre-processing (differentiable, before linear layers):**

    1. *Log-law wind profile* — ``WS_eff = WS · log(height/z0) / log(z_ref/z0)``
       with ``z0 = 0.01 m``, ``z_ref = 10.0 m``.
    2. *Ground albedo view factor* — ``VF = 0.5 · (1 − cos(tilt_rad))``,
       appended as an additional input feature alongside the 6 base inputs.

    **Architecture:** fully-connected tanh network.

    * ``tanh`` activations throughout — required for smooth gradients for
      PSO/BO optimizers.
    * Configurable depth (``n_layers``) and width (``hidden_dim``).
    * Output layer has no activation (raw regression).

    **Physical constraints (structural, not loss-based):**

    * ``T_panel ≥ T_amb`` at every timestep — enforced via ReLU clamp on the
      network's temperature delta output.
    * ``P_mp ≥ 0`` at every timestep — enforced via ReLU on the power output.
    """

    def __init__(
        self,
        n_layers: int = 6,
        hidden_dim: int = 128,
        n_timesteps: int = 96,
    ) -> None:
        """Build the surrogate.

        Args:
            n_layers: Number of hidden fully-connected layers.
            hidden_dim: Width of each hidden layer.
            n_timesteps: Number of output timesteps (default 96 → 15-min
                resolution over 24 h).
        """
        super().__init__()
        self.n_timesteps = n_timesteps
        self.hidden_dim = hidden_dim

        # Input projection: 7 features → hidden_dim.
        layers: list[nn.Module] = [nn.Linear(_SURROGATE_N_INPUTS, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)

        # Output projection: hidden_dim → T-delta (n_timesteps) + P (n_timesteps).
        self.output_proj = nn.Linear(hidden_dim, 2 * n_timesteps)

    # ------------------------------------------------------------------
    # Differentiable pre-processing helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _log_law(WS: Tensor, height: Tensor) -> Tensor:
        """Log-law effective wind speed at panel height (differentiable)."""
        return log_law_ws_eff(WS, height)

    @staticmethod
    def _view_factor(tilt: Tensor) -> Tensor:
        """Ground albedo view factor from tilt angle (differentiable).

        ``VF = 0.5 · (1 − cos(tilt_rad))``.
        """
        tilt_rad = tilt * (math.pi / 180.0)
        return 0.5 * (1.0 - torch.cos(tilt_rad))

    # ------------------------------------------------------------------
    # Forward pass.
    # ------------------------------------------------------------------

    def forward(
        self,
        G: Tensor,
        T_amb: Tensor,
        WS: Tensor,
        tilt: Tensor,
        azimuth: Tensor,
        height: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run a forward pass.

        Args:
            G: Plane-of-array irradiance [W/m²], shape ``(B,)``.
            T_amb: Ambient temperature [°C], shape ``(B,)``.
            WS: Wind speed at reference height [m/s], shape ``(B,)``.
            tilt: Panel tilt angle [degrees], shape ``(B,)``.
            azimuth: Panel azimuth angle [degrees], shape ``(B,)``.
            height: Panel mounting height [m], shape ``(B,)``.

        Returns:
            Tuple ``(T_panel, P_mp)`` where both tensors have shape
            ``(B, n_timesteps)``.

            * ``T_panel ≥ T_amb`` is structurally enforced at every timestep.
            * ``P_mp ≥ 0`` is structurally enforced at every timestep.
        """
        # --- differentiable pre-processing -------------------------------
        WS_eff = self._log_law(WS, height)           # (B,)
        VF = self._view_factor(tilt)                  # (B,)

        # Stack all 7 input features: (B, 7).
        features = torch.stack([G, T_amb, WS_eff, tilt, azimuth, height, VF], dim=-1)

        # --- network forward pass ----------------------------------------
        h = self.trunk(features)                      # (B, hidden_dim)
        raw = self.output_proj(h)                     # (B, 2·n_timesteps)

        raw_T_delta = raw[:, : self.n_timesteps]      # (B, n_timesteps)
        raw_P = raw[:, self.n_timesteps :]            # (B, n_timesteps)

        # --- structural physical constraints -----------------------------
        # T_panel ≥ T_amb: add T_amb broadcast + relu(delta).
        T_panel = T_amb.unsqueeze(-1) + F.relu(raw_T_delta)   # (B, n_timesteps)
        # P_mp ≥ 0: relu.
        P_mp = F.relu(raw_P)                                   # (B, n_timesteps)

        return T_panel, P_mp
