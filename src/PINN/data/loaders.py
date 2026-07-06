"""RK4TRAN CSV data loading and PyTorch Dataset."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class RK4TRANSample:
    """Single RK4TRAN sample with weather, panel state, and outputs."""

    location: tuple[float, float, float]  # lat, lon, elevation
    time_components: tuple[int, ...]  # hour, day, month, year, etc.
    weather: dict[str, float]  # T_amb, wind_speed, humidity, irradiance, etc.
    panel_state: dict[str, float]  # pv_height, pitch, roll, yaw
    outputs: dict[str, float]  # T_operating, T_operating_sigma, eta, eta_sigma
    raw_dict: dict[str, float]  # Full raw row for reference


class RK4TRANDataset(Dataset):
    """PyTorch Dataset for RK4TRAN synthetic training data.

    Loads CSV files from RK4TRAN output (spacious, comfortable, cramped grids).
    Handles Monte Carlo uncertainty quantification (sigma fields).

    Expected CSV columns:
    - location (lat lon elev)
    - time (hour day month year, etc.)
    - Weather: T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure
    - Panel state: pv_height, pitch, roll, yaw
    - Outputs: T_operating, T_operating_sigma, eta, eta_sigma
    """

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
            # Parse location (lat lon elev)
            location_str = row.get("location", "").strip()
            if not location_str:
                return None
            loc_parts = location_str.split()
            if len(loc_parts) < 2:
                return None
            location = (float(loc_parts[0]), float(loc_parts[1]), float(loc_parts[2]) if len(loc_parts) > 2 else 0.0)

            # Parse time components
            time_str = row.get("time", "").strip()
            time_components: tuple[int, ...] = tuple()
            if time_str:
                time_parts = time_str.split()
                time_components = tuple(int(p) for p in time_parts if p)

            # Parse weather
            weather = {k: float(row.get(k, "0")) for k in self.WEATHER_FIELDS}

            # Parse panel state
            panel_state = {k: float(row.get(k, "0")) for k in self.PANEL_STATE_FIELDS}

            # Parse outputs
            outputs = {k: float(row.get(k, "0")) for k in self.OUTPUT_FIELDS}

            return RK4TRANSample(
                location=location,
                time_components=time_components,
                weather=weather,
                panel_state=panel_state,
                outputs=outputs,
                raw_dict={k: float(row[k]) if k not in ["location", "time"] else row[k] for k in row},
            )
        except (ValueError, IndexError, TypeError) as e:
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
        }
        time_lengths = {len(sample.time_components) for sample in self.samples}
        if time_lengths == {0}:
            return normalizer_data
        if len(time_lengths) != 1:
            raise ValueError(
                "RK4TRAN dataset contains inconsistent time vector lengths; "
                f"found lengths {sorted(time_lengths)}"
            )
        normalizer_data["time"] = np.asarray(
            [sample.time_components for sample in self.samples],
            dtype=np.float32,
        )
        return normalizer_data
    
    def get_input_dim(self) -> int:
        """Return the flattened model input dimension for this dataset."""
        if not self.samples:
            raise ValueError("Cannot infer input dimension from an empty RK4TRAN dataset")
        sample = self.samples[0]
        return (
            len(self.WEATHER_FIELDS)
            + len(self.PANEL_STATE_FIELDS)
            + len(sample.location)
            + len(sample.time_components)
        )
        
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get sample by index.

        Returns dict with keys:
        - weather: [T_amb, wind_speed, wind_dir, humidity, irradiance, cloud_cover, pressure]
        - panel_state: [pv_height, pitch, roll, yaw]
        - location: [lat, lon, elevation]
        - time: [hour, day, month, year, ...]
        - T_operating: scalar
        - T_operating_sigma: scalar
        - eta: scalar
        - eta_sigma: scalar
        """
        sample = self.samples[idx]

        # Convert to tensors
        weather_vec = torch.tensor(
            [sample.weather.get(k, 0.0) for k in self.WEATHER_FIELDS],
            dtype=torch.float32,
        )
        panel_vec = torch.tensor(
            [sample.panel_state.get(k, 0.0) for k in self.PANEL_STATE_FIELDS],
            dtype=torch.float32,
        )
        location_vec = torch.tensor(sample.location, dtype=torch.float32)
        time_vec = torch.tensor(sample.time_components, dtype=torch.float32) if sample.time_components else torch.zeros(0)

        T_operating = torch.tensor(sample.outputs.get("T_operating", 0.0), dtype=torch.float32)
        eta = torch.tensor(sample.outputs.get("eta", 0.0), dtype=torch.float32)

        result = {
            "weather": weather_vec,
            "panel_state": panel_vec,
            "location": location_vec,
            "time": time_vec,
            "T_operating": T_operating,
            "eta": eta,
        }

        if self.include_uncertainty:
            result["T_operating_sigma"] = torch.tensor(sample.outputs.get("T_operating_sigma", 0.0), dtype=torch.float32)
            result["eta_sigma"] = torch.tensor(sample.outputs.get("eta_sigma", 0.0), dtype=torch.float32)

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
