"""Monte Carlo uncertainty quantification handling for PINN training."""

from __future__ import annotations

import torch
from torch import Tensor


class UncertaintyProcessor:
    """Handler for MC uncertainty bounds in RK4TRAN outputs.

    RK4TRAN provides T_operating_sigma and eta_sigma which represent
    uncertainty bounds from MC sampling. This processor handles different
    strategies for incorporating these bounds into training.
    """

    def __init__(self, strategy: str = "weighted") -> None:
        """Initialize UQ processor.

        Args:
            strategy: How to use sigma fields in training:
                - "weighted": Use sigma for loss weighting (lower sigma = higher weight)
                - "augment": Add noise sampled from Normal(0, sigma)
                - "keep": Keep sigma as separate channel for PINN to learn
        """
        valid_strategies = {"weighted", "augment", "keep"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Choose from {valid_strategies}")
        self.strategy = strategy

    def process(self, sample: dict[str, Tensor], training: bool = True) -> dict[str, Tensor]:
        """Process sample with UQ strategy.

        Args:
            sample: Dict with T_operating_sigma, eta_sigma, etc.
            training: If False, return sample as-is (use for inference)

        Returns:
            Processed sample dict
        """
        if not training or self.strategy == "keep":
            # Keep sigma fields as-is for weighting or learning
            return sample

        if self.strategy == "weighted":
            # Create weight tensors from sigma (inverse: smaller sigma = higher weight)
            if "T_operating_sigma" in sample:
                sigma = sample["T_operating_sigma"]
                sample["T_weight"] = 1.0 / (sigma + 1e-6)  # Avoid division by zero

            if "eta_sigma" in sample:
                sigma = sample["eta_sigma"]
                sample["eta_weight"] = 1.0 / (sigma + 1e-6)

        elif self.strategy == "augment":
            # Augment targets with noise
            if "T_operating" in sample and "T_operating_sigma" in sample:
                noise = torch.randn_like(sample["T_operating"]) * sample["T_operating_sigma"]
                sample["T_operating"] = sample["T_operating"] + noise

            if "eta" in sample and "eta_sigma" in sample:
                noise = torch.randn_like(sample["eta"]) * sample["eta_sigma"]
                sample["eta"] = sample["eta"] + noise

        return sample

    def get_loss_weights(self, sample: dict[str, Tensor]) -> dict[str, Tensor]:
        """Extract loss weights from processed sample.

        Args:
            sample: Processed sample dict

        Returns:
            Dict with 'T_weight' and 'eta_weight' tensors
        """
        weights = {}
        if "T_weight" in sample:
            weights["T_operating"] = sample["T_weight"]
        if "eta_weight" in sample:
            weights["eta"] = sample["eta_weight"]
        return weights


class UQStats:
    """Accumulator for UQ statistics across batches."""

    def __init__(self) -> None:
        """Initialize stats accumulator."""
        self.T_sigma_values: list[float] = []
        self.eta_sigma_values: list[float] = []

    def update(self, batch: dict[str, Tensor]) -> None:
        """Update statistics from batch.

        Args:
            batch: Batch dict with sigma fields
        """
        if "T_operating_sigma" in batch:
            sigma = batch["T_operating_sigma"]
            self.T_sigma_values.extend(sigma.detach().cpu().numpy().flatten().tolist())

        if "eta_sigma" in batch:
            sigma = batch["eta_sigma"]
            self.eta_sigma_values.extend(sigma.detach().cpu().numpy().flatten().tolist())

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics.

        Returns:
            Dict with mean, std, min, max for each sigma field
        """
        import numpy as np

        summary = {}
        if self.T_sigma_values:
            vals = np.array(self.T_sigma_values)
            summary["T_operating_sigma"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        if self.eta_sigma_values:
            vals = np.array(self.eta_sigma_values)
            summary["eta_sigma"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        return summary
