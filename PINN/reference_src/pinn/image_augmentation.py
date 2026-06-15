"""Training-time image augmentations (Batch H.3).

All operations are performed on pre-normalised ``float32`` tensors in
``[0, 1]`` with shape ``(C, H, W)`` (or batched ``(B, C, H, W)``) so they
can be composed with :class:`src.pinn.image_pipeline.SkyImagePipeline`
before the image enters the model.

Only used during training; the orchestrator and eval paths never call
:class:`SkyImageAugmentor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


__all__ = ["SkyImageAugmentor"]


@dataclass
class SkyImageAugmentor:
    """Compose a small bank of sky-image augmentations.

    Each knob is independently toggleable: set its probability/amount to
    zero to skip the corresponding step. The augmentor is seeded with
    its own :class:`torch.Generator` so it never perturbs global RNG
    state — matching :mod:`src.pinn.data`'s convention.

    Attributes:
        max_rotation_deg: Maximum absolute rotation in degrees (uniform
            in ``[-d, d]``). Zero disables rotation.
        hflip_prob: Probability of a horizontal flip.
        brightness_factor: Maximum ± brightness shift. Zero disables.
        noise_sigma: Standard deviation of additive Gaussian noise. Zero
            disables.
        seed: Seed for the local :class:`torch.Generator`.
    """

    max_rotation_deg: float = 15.0
    hflip_prob: float = 0.5
    brightness_factor: float = 0.1
    noise_sigma: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        self._gen = torch.Generator().manual_seed(int(self.seed))

    # --- pure per-step helpers ------------------------------------------

    def random_rotation(
        self, img: torch.Tensor, max_degrees: float = 15.0
    ) -> torch.Tensor:
        """Rotate ``img`` by a uniform angle in ``[-max_degrees, max_degrees]``.

        Uses a bilinearly-sampled affine warp (pure-torch, no OpenCV) so
        the output is differentiable and on the same device as ``img``.
        """
        if max_degrees <= 0:
            return img
        has_batch = img.dim() == 4
        x = img if has_batch else img.unsqueeze(0)
        angle = (
            torch.rand((), generator=self._gen) * (2 * max_degrees) - max_degrees
        )
        theta = float(angle) * torch.pi / 180.0
        cos, sin = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
        # 2x3 affine for a rotation about the image centre.
        matrix = torch.tensor(
            [[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=x.dtype, device=x.device
        ).unsqueeze(0).expand(x.shape[0], -1, -1)
        grid = torch.nn.functional.affine_grid(
            matrix, list(x.shape), align_corners=False
        )
        out = torch.nn.functional.grid_sample(
            x, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return out if has_batch else out.squeeze(0)

    def random_horizontal_flip(
        self, img: torch.Tensor, p: float = 0.5
    ) -> torch.Tensor:
        """Flip ``img`` left-right with probability ``p``."""
        if p <= 0:
            return img
        if float(torch.rand((), generator=self._gen)) < p:
            return torch.flip(img, dims=[-1])
        return img

    def random_brightness_jitter(
        self, img: torch.Tensor, factor: float = 0.1
    ) -> torch.Tensor:
        """Additive ± brightness shift, clamped to ``[0, 1]``.

        Operates on the pre-normalisation tensor; callers that feed an
        already-normalised tensor should disable this step.
        """
        if factor <= 0:
            return img
        shift = float(
            torch.rand((), generator=self._gen) * (2 * factor) - factor
        )
        return torch.clamp(img + shift, 0.0, 1.0)

    def random_gaussian_noise(
        self, img: torch.Tensor, sigma: float = 0.01
    ) -> torch.Tensor:
        """Add i.i.d. Gaussian noise of stddev ``sigma``, clamped to ``[0, 1]``."""
        if sigma <= 0:
            return img
        noise = torch.randn(img.shape, generator=self._gen, dtype=img.dtype) * sigma
        return torch.clamp(img + noise.to(img.device), 0.0, 1.0)

    # --- composition ----------------------------------------------------

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply every enabled augmentation in order."""
        img = self.random_rotation(img, self.max_rotation_deg)
        img = self.random_horizontal_flip(img, self.hflip_prob)
        img = self.random_brightness_jitter(img, self.brightness_factor)
        img = self.random_gaussian_noise(img, self.noise_sigma)
        return img
