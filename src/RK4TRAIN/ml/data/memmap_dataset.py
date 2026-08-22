"""Memory-mapped RK4TRAN dataset for files too large to load into RAM.

RK4TRANDataset (loaders.py) loads every row into a Python list -- fine for
smoke-test CSVs, completely unworkable once the lattice generator produces
files in the hundreds-of-GB to multi-TB range (confirmed necessary: Tommy's
full run at 30min hour-spacing is ~3.8 billion rows / ~2TB as CSV text).

This module converts a CSV (or list of CSVs) into a single flat binary
float32 file via `np.memmap`, then provides a PyTorch Dataset that reads
individual rows on demand. Because it's memory-mapped, the OS only pages in
the small slice actually being read for `__getitem__` -- RAM usage stays
minimal (a few MB) regardless of whether the underlying file is 1GB or 2TB.
Binary float32 storage is also considerably smaller than CSV text (roughly
35 cols x 4 bytes = 140 bytes/row here, vs ~525 bytes/row as ASCII text --
expect the converted file to be roughly 1/4 the size of the source CSV).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

# Canonical column order, matching RK4TRAN's actual CSV header exactly.
# Kept as a flat list (not the loaders.py field-group dicts) since the
# memmap's on-disk layout is one row = one contiguous array of all 35
# columns in this fixed order -- column *position*, not name lookup, is
# what makes random-access reads fast.
CSV_COLUMNS = [
    "lon", "lat", "alt", "minute", "hour", "day_of_year", "month", "year",
    "T_amb", "wind_speed", "wind_dir", "humidity", "irradiance", "cloud_cover", "pressure", "pv_height",
    "pitch", "roll", "yaw", "T_operating", "T_operating_sigma", "eta", "eta_sigma",
    "optimal_pitch", "optimal_roll", "optimal_yaw", "pitch_error", "roll_error", "yaw_error", "orientation_error",
    "T_panel_initial", "T_after_15min", "T_after_15min_sigma", "eta_after_15min", "eta_after_15min_sigma",
]
_COL_IDX = {name: i for i, name in enumerate(CSV_COLUMNS)}

LOCATION_FIELDS = ("lon", "lat", "alt")
TIME_FIELDS = ("minute", "hour", "day_of_year", "month", "year")
WEATHER_FIELDS = ("T_amb", "wind_speed", "wind_dir", "humidity", "irradiance", "cloud_cover", "pressure")
PANEL_STATE_FIELDS = ("pv_height", "pitch", "roll", "yaw")
PANEL_TEMP_FIELDS = ("T_panel_initial",)
OUTPUT_FIELDS = ("T_operating", "T_operating_sigma", "eta", "eta_sigma")
TRANSIENT_OUTPUT_FIELDS = ("T_after_15min", "T_after_15min_sigma", "eta_after_15min", "eta_after_15min_sigma")


def _count_rows_fast(csv_path: Path) -> int:
    """Count data rows (excluding header) via raw newline counting -- no
    parsing, so this is I/O-bound only and fast even on multi-hundred-GB
    files (reads in large binary chunks, counts b'\\n' occurrences)."""
    n = 0
    chunk_size = 64 * 1024 * 1024  # 64MB
    with open(csv_path, "rb") as f:
        header = f.readline()  # consume header, not counted
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            n += chunk.count(b"\n")
        # last line may lack a trailing newline
        f.seek(0, 2)
        if f.tell() > len(header):
            f.seek(-1, 2)
            if f.read(1) != b"\n":
                n += 1
    return n


def build_memmap(
    csv_paths: list[Path] | Path,
    out_dir: Path,
    name: str = "rk4tran",
    chunk_rows: int = 500_000,
) -> Path:
    """Convert one or more RK4TRAN CSVs into a single memory-mapped binary
    dataset. Streams through the input in chunks -- never holds more than
    `chunk_rows` rows in memory at once, regardless of total file size.

    Args:
        csv_paths: path(s) to RK4TRAN CSV file(s) to convert
        out_dir: directory to write the .dat (binary) and .meta.json files
        name: base filename for the output (out_dir/{name}.dat, {name}.meta.json)
        chunk_rows: rows per streaming read/write chunk

    Returns:
        Path to the .meta.json file (pass this, or out_dir, to RK4TRANMemmapDataset)
    """
    import pandas as pd  # local import: only needed for this conversion step

    if isinstance(csv_paths, Path):
        csv_paths = [csv_paths]
    csv_paths = [Path(p) for p in csv_paths]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Counting rows (I/O-bound pass, no parsing)...")
    row_counts = []
    for p in csv_paths:
        n = _count_rows_fast(p)
        row_counts.append(n)
        print(f"  {p.name}: {n:,} rows")
    total_rows = sum(row_counts)
    n_cols = len(CSV_COLUMNS)
    print(f"Total: {total_rows:,} rows x {n_cols} cols -> "
          f"{total_rows * n_cols * 4 / 1e9:.1f} GB binary (float32)")

    dat_path = out_dir / f"{name}.dat"
    meta_path = out_dir / f"{name}.meta.json"

    mmap = np.memmap(dat_path, dtype=np.float32, mode="w+", shape=(total_rows, n_cols))

    write_offset = 0
    for p in csv_paths:
        print(f"Converting {p.name}...")
        for chunk in pd.read_csv(p, chunksize=chunk_rows):
            # Reindex to the canonical column order regardless of the source
            # file's actual column order -- robust to schema drift across runs.
            chunk = chunk.reindex(columns=CSV_COLUMNS)
            arr = chunk.to_numpy(dtype=np.float32, na_value=0.0)
            n = arr.shape[0]
            mmap[write_offset : write_offset + n] = arr
            write_offset += n
    mmap.flush()
    assert write_offset == total_rows, f"row count mismatch: wrote {write_offset}, expected {total_rows}"

    meta = {
        "n_rows": total_rows,
        "n_cols": n_cols,
        "columns": CSV_COLUMNS,
        "dtype": "float32",
        "source_files": [str(p) for p in csv_paths],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done. Binary: {dat_path} ({dat_path.stat().st_size / 1e9:.1f} GB), meta: {meta_path}")
    return meta_path


class RK4TRANMemmapDataset(Dataset):
    """PyTorch Dataset backed by a memory-mapped binary file (see build_memmap).

    Random-access __getitem__ is O(1) and touches only the ~140 bytes for
    that one row -- RAM usage does not scale with dataset size. Use this
    instead of RK4TRANDataset (loaders.py) for any file too large to
    comfortably fit in RAM as a Python list (roughly: more than a few
    million rows, or whenever loaders.py's in-memory approach OOMs).
    """

    def __init__(
        self,
        meta_path: Path,
        normalizer: Optional[object] = None,
        include_uncertainty: bool = True,
    ) -> None:
        self.meta_path = Path(meta_path)
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)
        if self.meta["columns"] != CSV_COLUMNS:
            raise ValueError(
                "Memmap file's column schema doesn't match this module's CSV_COLUMNS -- "
                "was it built with an older version of build_memmap()? Rebuild it."
            )
        self.n_rows = self.meta["n_rows"]
        self.n_cols = self.meta["n_cols"]
        dat_path = self.meta_path.parent / self.meta_path.name.replace(".meta.json", ".dat")
        self._mmap = np.memmap(dat_path, dtype=np.float32, mode="r", shape=(self.n_rows, self.n_cols))
        self.normalizer = normalizer
        self.include_uncertainty = include_uncertainty

        self._loc_idx = [_COL_IDX[k] for k in LOCATION_FIELDS]
        self._time_idx = [_COL_IDX[k] for k in TIME_FIELDS]
        self._weather_idx = [_COL_IDX[k] for k in WEATHER_FIELDS]
        self._panel_idx = [_COL_IDX[k] for k in PANEL_STATE_FIELDS]
        self._panel_temp_idx = [_COL_IDX[k] for k in PANEL_TEMP_FIELDS]
        self._output_idx = {k: _COL_IDX[k] for k in OUTPUT_FIELDS}
        self._transient_idx = {k: _COL_IDX[k] for k in TRANSIENT_OUTPUT_FIELDS}

    def __len__(self) -> int:
        return self.n_rows

    def get_input_dim(self) -> int:
        return (
            len(WEATHER_FIELDS) + len(PANEL_STATE_FIELDS) + len(LOCATION_FIELDS)
            + len(TIME_FIELDS) + len(PANEL_TEMP_FIELDS)
        )

    def get_normalizer_data(self, sample_rows: Optional[int] = 5_000_000) -> dict[str, NDArray[np.float32]]:
        """Collect per-column min/max source data for fitting a normalizer.

        For datasets this large, fitting on a large random SAMPLE rather
        than the full file is the practical default (sample_rows=5M) --
        still representative for min/max purposes, avoids a full linear
        scan of a multi-hundred-GB+ file. Pass sample_rows=None to scan
        every row instead (slower, I/O-bound, but exact).
        """
        if sample_rows is not None and sample_rows < self.n_rows:
            rng = np.random.default_rng(seed=42)
            idx = np.sort(rng.choice(self.n_rows, size=sample_rows, replace=False))
            block = self._mmap[idx]
        else:
            block = self._mmap[:]  # full scan -- only use for smaller files
        return {
            "weather": np.asarray(block[:, self._weather_idx], dtype=np.float32),
            "panel_state": np.asarray(block[:, self._panel_idx], dtype=np.float32),
            "location": np.asarray(block[:, self._loc_idx], dtype=np.float32),
            "time": np.asarray(block[:, self._time_idx], dtype=np.float32),
            "panel_temp": np.asarray(block[:, self._panel_temp_idx], dtype=np.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = np.asarray(self._mmap[idx])  # copies just this one row out of the mapped file

        weather_vec = torch.tensor(row[self._weather_idx], dtype=torch.float32)
        panel_vec = torch.tensor(row[self._panel_idx], dtype=torch.float32)
        location_vec = torch.tensor(row[self._loc_idx], dtype=torch.float32)
        time_vec = torch.tensor(row[self._time_idx], dtype=torch.float32)
        panel_temp_vec = torch.tensor(row[self._panel_temp_idx], dtype=torch.float32)

        result = {
            "weather": weather_vec,
            "panel_state": panel_vec,
            "location": location_vec,
            "time": time_vec,
            "panel_temp": panel_temp_vec,
            "T_operating": torch.tensor(row[self._output_idx["T_operating"]], dtype=torch.float32),
            "eta": torch.tensor(row[self._output_idx["eta"]], dtype=torch.float32),
            "T_after_15min": torch.tensor(row[self._transient_idx["T_after_15min"]], dtype=torch.float32),
            "eta_after_15min": torch.tensor(row[self._transient_idx["eta_after_15min"]], dtype=torch.float32),
        }
        if self.include_uncertainty:
            result["T_operating_sigma"] = torch.tensor(row[self._output_idx["T_operating_sigma"]], dtype=torch.float32)
            result["eta_sigma"] = torch.tensor(row[self._output_idx["eta_sigma"]], dtype=torch.float32)
            result["T_after_15min_sigma"] = torch.tensor(row[self._transient_idx["T_after_15min_sigma"]], dtype=torch.float32)
            result["eta_after_15min_sigma"] = torch.tensor(row[self._transient_idx["eta_after_15min_sigma"]], dtype=torch.float32)

        if self.normalizer:
            result = self.normalizer.normalize(result)
        return result

    def batch_iterator(self, batch_size: int, shuffle: bool = True, seed: Optional[int] = None):
        """Yields batched sample dicts directly, bypassing DataLoader's default
        per-index __getitem__ + collate path entirely.

        WHY THIS EXISTS (see checklist Section 2 benchmark notes): the standard
        DataLoader(dataset, batch_size=N) approach calls __getitem__ once per
        row -- for each row, that's 4-6 separate torch.tensor() constructions
        plus dict packing, all in a Python loop. Measured directly at 1.76M
        rows: ~0.22ms/row for __getitem__ ALONE (no model compute) -- which
        extrapolates to ~2.3 HOURS just for data loading on a full 38M-row
        location, per epoch, before the network even runs. That's a classic,
        well-known anti-pattern for tabular PyTorch datasets, not a fundamental
        compute limit -- increasing batch_size or DataLoader num_workers
        doesn't fix it (confirmed: neither helped in testing), because the
        per-row Python overhead dominates regardless of how the resulting
        tensors get grouped or which process runs the loop.

        This method instead does ONE vectorized numpy fancy-index per batch
        (pulling all B rows at once) and ONE torch.from_numpy() conversion per
        field group (weather/panel_state/location/time/panel_temp), not per
        row. Batch normalization reuses NumericNormalizer.normalize()
        unchanged -- it already broadcasts correctly over a batch leading
        dimension.

        Args:
            batch_size: rows per yielded batch
            shuffle: shuffle row order each call (recommended for training)
            seed: RNG seed for the shuffle, for reproducibility

        Yields:
            dict of batched tensors, same keys as __getitem__ but each
            tensor has a leading batch dimension [B, ...] instead of being
            a single unbatched row.
        """
        n = self.n_rows
        rng = np.random.default_rng(seed)
        order = rng.permutation(n) if shuffle else np.arange(n)

        for start in range(0, n, batch_size):
            idx_batch = order[start:start + batch_size]
            idx_sorted = np.sort(idx_batch)  # memmap fancy-indexing is faster with sorted indices
            rows = np.asarray(self._mmap[idx_sorted])  # ONE vectorized read for the whole batch

            batch = {
                "weather": torch.from_numpy(rows[:, self._weather_idx].copy()),
                "panel_state": torch.from_numpy(rows[:, self._panel_idx].copy()),
                "location": torch.from_numpy(rows[:, self._loc_idx].copy()),
                "time": torch.from_numpy(rows[:, self._time_idx].copy()),
                "panel_temp": torch.from_numpy(rows[:, self._panel_temp_idx].copy()),
                "T_operating": torch.from_numpy(rows[:, self._output_idx["T_operating"]].copy()),
                "eta": torch.from_numpy(rows[:, self._output_idx["eta"]].copy()),
                "T_after_15min": torch.from_numpy(rows[:, self._transient_idx["T_after_15min"]].copy()),
                "eta_after_15min": torch.from_numpy(rows[:, self._transient_idx["eta_after_15min"]].copy()),
            }
            if self.include_uncertainty:
                batch["T_operating_sigma"] = torch.from_numpy(rows[:, self._output_idx["T_operating_sigma"]].copy())
                batch["eta_sigma"] = torch.from_numpy(rows[:, self._output_idx["eta_sigma"]].copy())
                batch["T_after_15min_sigma"] = torch.from_numpy(rows[:, self._transient_idx["T_after_15min_sigma"]].copy())
                batch["eta_after_15min_sigma"] = torch.from_numpy(rows[:, self._transient_idx["eta_after_15min_sigma"]].copy())

            if self.normalizer:
                batch = self.normalizer.normalize(batch)
            yield batch
