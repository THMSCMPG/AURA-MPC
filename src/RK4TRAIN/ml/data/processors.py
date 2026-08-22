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

    Stores min/max PER COLUMN within each field group (e.g. per-weather-
    variable, not one shared range across all 7 weather columns) and applies
    normalization/denormalization. Supports save/load for reproducibility.

    BUGFIX (found reviewing the training pipeline): an earlier version fit
    and applied a single (min,max) pair per *group* (e.g. "weather"), by
    flattening all columns in that group together. Since weather mixes
    pressure (~50,000-105,000 Pa) with humidity/cloud_cover (0-1) and T_amb
    (~233-333 K), that single shared range let pressure's scale dominate,
    squashing nearly every other weather feature into a sliver near 0 after
    normalization -- a real problem for training. Fixed by tracking one
    (min,max) per column instead.
    """

    def __init__(self, field_ranges: dict[str, list[tuple[float, float]]] | None = None) -> None:
        """Initialize normalizer with optional predefined ranges.

        Args:
            field_ranges: Dict mapping field/group name to a list of
                (min, max) tuples, one per column in that group.
        """
        self.field_ranges: dict[str, list[tuple[float, float]]] = field_ranges or {}
        self.fitted = bool(field_ranges)

    def fit(self, data: dict[str, NDArray]) -> None:
        """Fit normalizer on data.

        Args:
            data: Dict mapping field/group names to 2D arrays shaped [N, D]
                (D columns per group, e.g. weather's 7 variables). A 1D
                array is treated as D=1 (single column).
        """
        self.field_ranges = {}
        for field, values in data.items():
            values_arr = np.asarray(values)
            if values_arr.size == 0:
                raise ValueError(f"Cannot fit normalizer on empty field '{field}'")
            if values_arr.ndim == 1:
                values_arr = values_arr.reshape(-1, 1)
            mins = np.min(values_arr, axis=0)
            maxs = np.max(values_arr, axis=0)
            self.field_ranges[field] = [(float(lo), float(hi)) for lo, hi in zip(mins, maxs)]
        self.fitted = True

    def normalize_field(self, field: str, value: float, column: int = 0) -> float:
        """Normalize a single scalar value against one column's range.

        Args:
            field: Field/group name
            value: Raw value
            column: Which column within the group's range list to use
                (default 0, for single-column fields)

        Returns:
            Normalized value in [0, 1]
        """
        if field not in self.field_ranges:
            return value
        min_val, max_val = self.field_ranges[field][column]
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def denormalize_field(self, field: str, value: float, column: int = 0) -> float:
        """Denormalize a single scalar value against one column's range.

        Args:
            field: Field/group name
            value: Normalized value
            column: Which column within the group's range list to use

        Returns:
            Raw value
        """
        if field not in self.field_ranges:
            return value
        min_val, max_val = self.field_ranges[field][column]
        return value * (max_val - min_val) + min_val

    def normalize(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize sample dict, per-column within each group.

        Args:
            sample: Dict with keys like 'weather', 'panel_state', etc.,
                each tensor's last dimension matching that group's column
                count (e.g. weather: [..., 7]).

        Returns:
            Normalized sample dict
        """
        normalized = {}
        for key, tensor in sample.items():
            if key in self.field_ranges:
                ranges = self.field_ranges[key]
                if tensor.shape[-1] != len(ranges):
                    raise ValueError(
                        f"Normalizer field '{key}' has {len(ranges)} fitted columns but "
                        f"the tensor's last dimension is {tensor.shape[-1]} -- schema mismatch, "
                        "likely fit() was called on data with a different column layout."
                    )
                mins = torch.tensor([r[0] for r in ranges], dtype=tensor.dtype, device=tensor.device)
                maxs = torch.tensor([r[1] for r in ranges], dtype=tensor.dtype, device=tensor.device)
                span = maxs - mins
                span = torch.where(span == 0, torch.ones_like(span), span)  # avoid div-by-zero per column
                normalized[key] = torch.where(
                    (maxs - mins).eq(0),
                    torch.zeros_like(tensor),
                    (tensor - mins) / span,
                )
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
            raw = json.load(f)
        # JSON round-trips tuples as lists; convert back to tuples for consistency.
        field_ranges = {field: [tuple(r) for r in ranges] for field, ranges in raw.items()}
        return cls(field_ranges=field_ranges)

    def get_ranges(self) -> dict[str, list[tuple[float, float]]]:
        """Get field ranges (list of (min,max) per column, keyed by group name)."""
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
