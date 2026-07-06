"""Data preprocessing and normalization utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray


class NumericNormalizer:
    """Min-max normalizer for numeric fields.

    Stores min/max for each field and applies normalization/denormalization.
    Supports save/load for reproducibility.
    """

    def __init__(self, field_ranges: dict[str, tuple[float, float]] | None = None) -> None:
        """Initialize normalizer with optional predefined ranges.

        Args:
            field_ranges: Dict mapping field name to (min, max) tuple
        """
        self.field_ranges = field_ranges or {}
        self.fitted = bool(field_ranges)

    def fit(self, data: dict[str, NDArray]) -> None:
        """Fit normalizer on data.

        Args:
            data: Dict mapping field names to arrays of values
        """
        self.field_ranges = {}
        for field, values in data.items():
            values_arr = np.array(values).flatten()
            self.field_ranges[field] = (float(np.min(values_arr)), float(np.max(values_arr)))
        self.fitted = True

    def normalize_field(self, field: str, value: float) -> float:
        """Normalize single value.

        Args:
            field: Field name
            value: Raw value

        Returns:
            Normalized value in [0, 1]
        """
        if field not in self.field_ranges:
            return value
        min_val, max_val = self.field_ranges[field]
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def denormalize_field(self, field: str, value: float) -> float:
        """Denormalize single value.

        Args:
            field: Field name
            value: Normalized value

        Returns:
            Raw value
        """
        if field not in self.field_ranges:
            return value
        min_val, max_val = self.field_ranges[field]
        return value * (max_val - min_val) + min_val

    def normalize(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize sample dict.

        Args:
            sample: Dict with keys like 'weather', 'panel_state', etc.

        Returns:
            Normalized sample dict
        """
        normalized = {}
        for key, tensor in sample.items():
            if key in self.field_ranges:
                min_val, max_val = self.field_ranges[key]
                if max_val == min_val:
                    normalized[key] = torch.zeros_like(tensor)
                else:
                    normalized[key] = (tensor - min_val) / (max_val - min_val)
            else:
                normalized[key] = tensor
        return normalized

    def save(self, path: Path) -> None:
        """Save normalizer to JSON.

        Args:
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.field_ranges, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> NumericNormalizer:
        """Load normalizer from JSON.

        Args:
            path: Input file path

        Returns:
            Loaded NumericNormalizer instance
        """
        with open(path, "r") as f:
            field_ranges = json.load(f)
        return cls(field_ranges=field_ranges)

    def get_ranges(self) -> dict[str, tuple[float, float]]:
        """Get field ranges."""
        return self.field_ranges.copy()


class DataProcessor:
    """Unified data processing pipeline."""

    def __init__(
        self,
        normalizer: NumericNormalizer | None = None,
        uncertainty_processor: UncertaintyProcessor | None = None,
    ) -> None:
        """Initialize processor.

        Args:
            normalizer: Normalization handler
            uncertainty_processor: UQ handler
        """
        self.normalizer = normalizer
        self.uncertainty_processor = uncertainty_processor

    def process(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply full processing pipeline.

        Args:
            sample: Input sample dict

        Returns:
            Processed sample dict
        """
        if self.normalizer:
            sample = self.normalizer.normalize(sample)
        if self.uncertainty_processor:
            sample = self.uncertainty_processor.process(sample)
        return sample


# Placeholder for UQ processor
class UncertaintyProcessor:
    """Monte Carlo uncertainty quantification processor."""

    def __init__(self, strategy: str = "mean") -> None:
        """Initialize UQ processor.

        Args:
            strategy: How to handle sigma fields
                - "mean": use sigma for uncertainty weighting in loss
                - "sample": sample from Normal(value, sigma)
                - "keep": keep sigma as separate output
        """
        self.strategy = strategy

    def process(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Process sample with UQ.

        Args:
            sample: Input sample dict

        Returns:
            Processed sample dict
        """
        # For now, keep sigma fields as-is; they'll be used in loss weighting
        return sample
