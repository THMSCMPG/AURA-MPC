"""RK4TRAN CSV data loading and PyTorch Dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class RK4TRANSample:
    """Single RK4TRAN sample with weather, panel state, and outputs."""

    location: tuple[float, float, float]  # lon, lat, elevation -- canonical order, matches RK4TRAN's Fortran side
    time_components: tuple[float, ...]  # minute, hour, day_of_year, month, year
    weather: dict[str, float]  # T_amb, wind_speed, humidity, irradiance, etc.
    panel_state: dict[str, float]  # pv_height, pitch, roll, yaw
    outputs: dict[str, float]  # T_operating, T_operating_sigma, eta, eta_sigma (steady state)
    transient_outputs: dict[str, float] = field(default_factory=dict)  # T_after_15min, eta_after_15min, + sigmas
    panel_temp_state: dict[str, float] = field(default_factory=dict)  # T_panel_initial
    analysis: dict[str, float] = field(default_factory=dict)  # optimal_*, *_error columns -- RK4TRAN-only, not PINN I/O (see D9)
    raw_dict: dict[str, float] = field(default_factory=dict)  # Full raw row for reference


class RK4TRANDataset(Dataset):
    """PyTorch Dataset for RK4TRAN synthetic training data.

    Loads CSV files from RK4TRAN's full-factorial lattice generator.

    Expected CSV columns (current schema, separate numeric columns --
    NOT the old packed "location"/"time" strings):
    - Location: lon, lat, alt
    - Time: minute, hour, day_of_year, month, year
    - Weather: T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure
    - Panel state: pv_height, pitch, roll, yaw
    - Steady-state outputs: T_operating, T_operating_sigma, eta, eta_sigma
    - Transient (15-min) outputs: T_after_15min, T_after_15min_sigma, eta_after_15min, eta_after_15min_sigma
    - Panel temperature state: T_panel_initial (independent axis, NOT the same as T_operating)
    - Analysis-only columns (not model I/O -- see D9 in the project checklist):
      optimal_pitch, optimal_roll, optimal_yaw, pitch_error, roll_error, yaw_error, orientation_error

    NOTE: transient_outputs, panel_temp_state, and analysis fields are parsed
    and available on every RK4TRANSample / in raw_dict, but are deliberately
    NOT wired into __getitem__'s default returned tensors -- whether the
    transient prediction becomes a new PINNSurrogate output head (expanding
    past the current 4 outputs) is an open architecture decision, not yet
    resolved. Wiring it in silently here would commit to that decision
    without a confirmed answer. See the project checklist, Section 2.
    """

    LOCATION_FIELDS = ("lon", "lat", "alt")
    TIME_FIELDS = ("minute", "hour", "day_of_year", "month", "year")
    WEATHER_FIELDS = (
        "T_amb",
        "wind_speed",
        "wind_dir",
        "humidity",
        "irradiance",
        "cloud_cover",
        "pressure",
    )
    PANEL_STATE_FIELDS = ("pv_height", "pitch", "roll", "yaw")
    OUTPUT_FIELDS = ("T_operating", "T_operating_sigma", "eta", "eta_sigma")
    TRANSIENT_OUTPUT_FIELDS = ("T_after_15min", "T_after_15min_sigma", "eta_after_15min", "eta_after_15min_sigma")
    PANEL_TEMP_FIELDS = ("T_panel_initial",)
    ANALYSIS_FIELDS = (
        "optimal_pitch", "optimal_roll", "optimal_yaw",
        "pitch_error", "roll_error", "yaw_error", "orientation_error",
    )

    def __init__(
        self,
        csv_paths: list[Path] | Path,
        normalizer: Optional[object] = None,
        include_uncertainty: bool = True,
    ) -> None:
        """Initialize RK4TRAN dataset.

        Args:
            csv_paths: Path(s) to RK4TRAN CSV file(s)
            normalizer: Optional normalizer for preprocessing
            include_uncertainty: If True, include sigma fields in outputs
        """
        if isinstance(csv_paths, Path):
            csv_paths = [csv_paths]
        self.csv_paths = [Path(p) for p in csv_paths]
        self.normalizer = normalizer
        self.include_uncertainty = include_uncertainty
        self.samples: list[RK4TRANSample] = []

        self._load_csv_files()

    def _load_csv_files(self) -> None:
        """Load and parse all CSV files."""
        for csv_path in self.csv_paths:
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV not found: {csv_path}")

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise ValueError(f"Empty CSV: {csv_path}")

                for row in reader:
                    try:
                        sample = self._parse_row(row)
                        if sample:
                            self.samples.append(sample)
                    except (ValueError, KeyError) as e:
                        print(f"Warning: skipping row {len(self.samples)}: {e}")
                        continue

    def _parse_row(self, row: dict[str, str]) -> Optional[RK4TRANSample]:
        """Parse single CSV row into RK4TRANSample."""
        try:
            if not row.get("lon") or not row.get("lat"):
                return None

            location = tuple(float(row[k]) for k in self.LOCATION_FIELDS)
            time_components = tuple(float(row.get(k, "0")) for k in self.TIME_FIELDS)
            weather = {k: float(row.get(k, "0")) for k in self.WEATHER_FIELDS}
            panel_state = {k: float(row.get(k, "0")) for k in self.PANEL_STATE_FIELDS}
            outputs = {k: float(row.get(k, "0")) for k in self.OUTPUT_FIELDS}

            # Optional groups -- older CSVs (pre-transient-prediction schema)
            # won't have these columns; parse only what's actually present.
            transient_outputs = {
                k: float(row[k]) for k in self.TRANSIENT_OUTPUT_FIELDS if k in row and row[k] != ""
            }
            panel_temp_state = {
                k: float(row[k]) for k in self.PANEL_TEMP_FIELDS if k in row and row[k] != ""
            }
            analysis = {
                k: float(row[k]) for k in self.ANALYSIS_FIELDS if k in row and row[k] != ""
            }

            skip_raw = set(self.LOCATION_FIELDS) | set(self.TIME_FIELDS)
            raw_dict = {}
            for k, v in row.items():
                if k in skip_raw:
                    continue
                try:
                    raw_dict[k] = float(v)
                except (ValueError, TypeError):
                    raw_dict[k] = v  # non-numeric field, keep as-is rather than dropping

            return RK4TRANSample(
                location=location,
                time_components=time_components,
                weather=weather,
                panel_state=panel_state,
                outputs=outputs,
                transient_outputs=transient_outputs,
                panel_temp_state=panel_temp_state,
                analysis=analysis,
                raw_dict=raw_dict,
            )
        except (ValueError, IndexError, TypeError, KeyError) as e:
            raise ValueError(f"Failed to parse row: {e}")

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.samples)

    def get_normalizer_data(self) -> dict[str, NDArray[np.float32]]:
        """Collect numeric feature arrays for fitting a normalizer.
        Returns:
            Dict of feature group name to 2D float32 array shaped [N, D]
        """
        if not self.samples:
            raise ValueError("Cannot fit normalizer on an empty RK4TRAN dataset")
        normalizer_data: dict[str, NDArray[np.float32]] = {
            "weather": np.asarray(
                [[sample.weather.get(k, 0.0) for k in self.WEATHER_FIELDS] for sample in self.samples],
                dtype=np.float32,
            ),
            "panel_state": np.asarray(
                [[sample.panel_state.get(k, 0.0) for k in self.PANEL_STATE_FIELDS] for sample in self.samples],
                dtype=np.float32,
            ),
            "location": np.asarray([sample.location for sample in self.samples], dtype=np.float32),
            "time": np.asarray([sample.time_components for sample in self.samples], dtype=np.float32),
            "panel_temp": np.asarray(
                [[sample.panel_temp_state.get(k, 0.0) for k in self.PANEL_TEMP_FIELDS] for sample in self.samples],
                dtype=np.float32,
            ),
        }
        return normalizer_data

    def get_input_dim(self) -> int:
        """Return the flattened model input dimension for this dataset."""
        if not self.samples:
            raise ValueError("Cannot infer input dimension from an empty RK4TRAN dataset")
        return (
            len(self.WEATHER_FIELDS)
            + len(self.PANEL_STATE_FIELDS)
            + len(self.LOCATION_FIELDS)
            + len(self.TIME_FIELDS)
            + len(self.PANEL_TEMP_FIELDS)
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample by index.

        Returns dict with keys:
        - weather: [T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure]
        - panel_state: [pv_height, pitch, roll, yaw]
        - location: [lon, lat, elevation]
        - time: [minute, hour, day_of_year, month, year]
        - panel_temp: [T_panel_initial] -- the starting temperature for the
          15-min transient prediction (see PINNSurrogate, D10). Required
          model input as of this schema -- NOT optional/analysis-only
          (unlike the optimal-orientation/error columns, see D9).
        - T_operating, eta, T_after_15min, eta_after_15min (+ sigmas if
          include_uncertainty) -- training targets. eta/eta_after_15min are
          present here for reference/validation, but PINNSurrogate does not
          learn them directly -- it derives them from the T predictions
          (see models.pinn.PINNSurrogate class docstring).

        The optimal-orientation/error analysis columns remain excluded (see
        D9) -- access them via `dataset.samples[idx]` if needed.
        """
        sample = self.samples[idx]

        weather_vec = torch.tensor(
            [sample.weather.get(k, 0.0) for k in self.WEATHER_FIELDS],
            dtype=torch.float32,
        )
        panel_vec = torch.tensor(
            [sample.panel_state.get(k, 0.0) for k in self.PANEL_STATE_FIELDS],
            dtype=torch.float32,
        )
        location_vec = torch.tensor(sample.location, dtype=torch.float32)
        time_vec = torch.tensor(sample.time_components, dtype=torch.float32)
        panel_temp_vec = torch.tensor(
            [sample.panel_temp_state.get(k, 0.0) for k in self.PANEL_TEMP_FIELDS],
            dtype=torch.float32,
        )

        T_operating = torch.tensor(sample.outputs.get("T_operating", 0.0), dtype=torch.float32)
        eta = torch.tensor(sample.outputs.get("eta", 0.0), dtype=torch.float32)
        T_after_15min = torch.tensor(sample.transient_outputs.get("T_after_15min", 0.0), dtype=torch.float32)
        eta_after_15min = torch.tensor(sample.transient_outputs.get("eta_after_15min", 0.0), dtype=torch.float32)

        result = {
            "weather": weather_vec,
            "panel_state": panel_vec,
            "location": location_vec,
            "time": time_vec,
            "panel_temp": panel_temp_vec,
            "T_operating": T_operating,
            "eta": eta,
            "T_after_15min": T_after_15min,
            "eta_after_15min": eta_after_15min,
        }

        if self.include_uncertainty:
            result["T_operating_sigma"] = torch.tensor(sample.outputs.get("T_operating_sigma", 0.0), dtype=torch.float32)
            result["eta_sigma"] = torch.tensor(sample.outputs.get("eta_sigma", 0.0), dtype=torch.float32)
            result["T_after_15min_sigma"] = torch.tensor(sample.transient_outputs.get("T_after_15min_sigma", 0.0), dtype=torch.float32)
            result["eta_after_15min_sigma"] = torch.tensor(sample.transient_outputs.get("eta_after_15min_sigma", 0.0), dtype=torch.float32)

        if self.normalizer:
            result = self.normalizer.normalize(result)

        return result


def create_dataloaders(
    csv_paths: list[Path],
    batch_size: int = 64,
    train_split: float = 0.8,
    val_split: float = 0.1,
    normalizer: Optional[object] = None,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from RK4TRAN CSV files.

    Args:
        csv_paths: List of CSV file paths
        batch_size: Batch size for dataloaders
        train_split: Fraction for training (default 0.8)
        val_split: Fraction for validation (default 0.1); test = 1 - train - val
        normalizer: Optional normalizer for preprocessing
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load full dataset
    dataset = RK4TRANDataset(csv_paths, normalizer=normalizer)

    # Split indices
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(dataset)
    indices = np.random.permutation(n)

    train_size = int(train_split * n)
    val_size = int(val_split * n)

    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    from torch.utils.data import Subset

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
