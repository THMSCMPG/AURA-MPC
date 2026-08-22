#!/usr/bin/env python3
"""optuna_pretrain.py -- Hyperparameter tuning for PINN offline pretraining.

Mirrors sandbox/optuna_tune.py's pattern (TPE sampler, median pruner, best-
config snapshot written to YAML) but targets Section 2's offline supervised
pretraining against RK4TRAN's lattice output, rather than the live RL loop.

Works with EITHER dataset backend:
  - RK4TRANMemmapDataset, for real-scale data (recommended -- required once
    files are large enough that RK4TRANDataset's in-memory list approach
    would OOM; build the memmap first with PINN/data/memmap_dataset.py's
    build_memmap()).
  - RK4TRANDataset, for small/smoke-test CSVs.

IMPORTANT for real-scale (multi-hundred-GB to multi-TB) data: hyperparameter
search trains many short trials, so this tunes against a random SUBSET of
the full dataset by default (--tune-subset-size, default 2,000,000 rows) --
not the entire multi-billion-row set. Once you have a winning config, run a
SEPARATE full training pass (plain run_pretrain.py-style, no Optuna) against
the complete dataset. Tuning against the full set every trial would be
prohibitively slow for no real HPO benefit -- relative hyperparameter
rankings are stable across a large-enough random subset.

Usage:
  # after building a memmap from the real RK4TRAN output:
  python src/RK4TRAIN/ml/scripts/optuna_pretrain.py \\
    --memmap-meta src/RK4TRAIN/lattice_batches/memmap/rk4tran.meta.json \\
    --trials 40 --epochs-per-trial 8 --tune-subset-size 2000000 \\
    --study-name aura_pretrain_tune --storage sqlite:///optuna_pretrain.db
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

THIS_FILE = Path(__file__).resolve()
ML_DIR = THIS_FILE.parents[1]  # .../src/RK4TRAIN/ml
SRC_DIR = ML_DIR.parent.parent  # .../src

for p in (str(SRC_DIR), str(ML_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data import NumericNormalizer, RK4TRANDataset
from data.memmap_dataset import RK4TRANMemmapDataset
from models import PINNSurrogate
from training import Trainer


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def build_dataset(args: argparse.Namespace) -> Dataset:
    """Build the underlying (un-split, un-normalized) dataset from either backend."""
    if args.memmap_meta is not None:
        print(f"Loading memmap dataset from {args.memmap_meta}")
        ds = RK4TRANMemmapDataset(args.memmap_meta)
        print(f"  {len(ds):,} rows (memory-mapped -- not loaded into RAM)")
        return ds
    if args.csv_dir is not None:
        csv_files = sorted(Path(args.csv_dir).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {args.csv_dir}")
        print(f"Loading {len(csv_files)} CSV file(s) from {args.csv_dir} (in-memory -- only use for small files)")
        ds = RK4TRANDataset(csv_files)
        print(f"  {len(ds):,} rows loaded")
        return ds
    raise ValueError("Must provide either --memmap-meta or --csv-dir")


def split_indices(n: int, train_split: float, val_split: float, seed: int) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    return indices[:train_size], indices[train_size : train_size + val_size], indices[train_size + val_size :]


NDArrayInt = np.ndarray  # alias for the type hint above (avoids importing NDArray[np.int_] verbosity)


def make_loaders(
    dataset: Dataset,
    normalizer: Optional[NumericNormalizer],
    batch_size: int,
    train_split: float,
    val_split: float,
    seed: int,
    num_workers: int,
    subset_size: Optional[int] = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders. If subset_size is set, first restricts to
    a random subset of that many rows (for fast HPO trials), THEN splits
    that subset into train/val -- keeps each trial's data volume bounded
    regardless of the underlying dataset's real size."""
    n_total = len(dataset)  # type: ignore[arg-type]
    if subset_size is not None and subset_size < n_total:
        rng = np.random.default_rng(seed)
        subset_idx = rng.choice(n_total, size=subset_size, replace=False)
        dataset = Subset(dataset, subset_idx.tolist())
        n_total = subset_size

    train_idx, val_idx, _test_idx = split_indices(n_total, train_split, val_split, seed)

    class _Wrapped(Dataset):
        def __init__(self, base: Dataset, idx: NDArrayInt) -> None:
            self.base = base
            self.idx = idx

        def __len__(self) -> int:
            return len(self.idx)

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            item = self.base[int(self.idx[i])]
            if normalizer is not None:
                item = normalizer.normalize(item)
            return item

    train_loader = DataLoader(_Wrapped(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(_Wrapped(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def get_input_dim(dataset: Dataset) -> int:
    if isinstance(dataset, (RK4TRANMemmapDataset, RK4TRANDataset)):
        return dataset.get_input_dim()
    raise TypeError(f"Unknown dataset type for input_dim: {type(dataset)}")


def make_trial_config(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "model": {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
            "num_residual_blocks": trial.suggest_int("num_residual_blocks", 2, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        },
        "training": {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        },
        "data": {
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        },
    }


def run_trial(
    cfg: dict[str, Any],
    dataset: Dataset,
    normalizer: Optional[NumericNormalizer],
    input_dim: int,
    epochs: int,
    seed: int,
    device: str,
    num_workers: int,
    subset_size: Optional[int],
    trial: optuna.Trial,
) -> float:
    train_loader, val_loader = make_loaders(
        dataset, normalizer,
        batch_size=cfg["data"]["batch_size"],
        train_split=0.8, val_split=0.2,  # no held-out test split needed during HPO
        seed=seed, num_workers=num_workers, subset_size=subset_size,
    )

    model = PINNSurrogate(
        input_dim=input_dim,
        hidden_dim=cfg["model"]["hidden_dim"],
        num_residual_blocks=cfg["model"]["num_residual_blocks"],
        num_outputs=4,
        dropout=cfg["model"]["dropout"],
    )
    trainer = Trainer(model=model, device=device, lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    best_val = float("inf")
    for epoch in range(epochs):
        trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics["val_loss"]
        best_val = min(best_val, val_loss)

        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-meta", type=Path, default=None, help=".meta.json from build_memmap() -- use for real-scale data")
    p.add_argument("--csv-dir", type=Path, default=None, help="Directory of CSVs -- only for small/smoke-test data")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--epochs-per-trial", type=int, default=8, help="Kept short deliberately -- this ranks configs, doesn't train the final model")
    p.add_argument("--tune-subset-size", type=int, default=2_000_000, help="Random subset size per trial, for HPO speed on huge datasets. Set to a very large number (or 0) to disable and use the full dataset -- NOT recommended for multi-billion-row data.")
    p.add_argument("--normalizer-sample-rows", type=int, default=5_000_000, help="Rows sampled to fit the normalizer (only matters for RK4TRANMemmapDataset)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--study-name", type=str, default="aura_pretrain_tune")
    p.add_argument("--storage", type=str, default="sqlite:///optuna_pretrain.db")
    p.add_argument("--out-dir", type=Path, default=Path("src/RK4TRAIN/ml/outputs/optuna"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.memmap_meta is None and args.csv_dir is None:
        sys.exit("Must provide --memmap-meta (real-scale data) or --csv-dir (small test data)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    input_dim = get_input_dim(dataset)
    print(f"input_dim: {input_dim}")

    normalizer = None
    if isinstance(dataset, RK4TRANMemmapDataset):
        print(f"Fitting normalizer on a {args.normalizer_sample_rows:,}-row sample...")
        normalizer = NumericNormalizer()
        normalizer.fit(dataset.get_normalizer_data(sample_rows=args.normalizer_sample_rows))
    else:
        print("Fitting normalizer on full (small) dataset...")
        normalizer = NumericNormalizer()
        normalizer.fit(dataset.get_normalizer_data())
    normalizer.save(args.out_dir / "normalizer.json")

    subset_size = args.tune_subset_size if args.tune_subset_size > 0 else None

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = make_trial_config(trial)
        return run_trial(
            cfg, dataset, normalizer, input_dim,
            epochs=args.epochs_per_trial, seed=args.seed, device=args.device,
            num_workers=args.num_workers, subset_size=subset_size, trial=trial,
        )

    study.optimize(objective, n_trials=args.trials)

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"number: {best.number}")
    print(f"val_loss: {best.value:.6f}")
    print("params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    best_cfg = {
        "model": {
            "input_dim": input_dim,
            "hidden_dim": best.params["hidden_dim"],
            "num_residual_blocks": best.params["num_residual_blocks"],
            "num_outputs": 4,
            "dropout": best.params["dropout"],
        },
        "training": {
            "learning_rate": best.params["learning_rate"],
            "weight_decay": best.params["weight_decay"],
        },
        "data": {
            "batch_size": best.params["batch_size"],
        },
    }
    best_cfg_path = args.out_dir / "best_pretrain.yaml"
    with best_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"\nBest config written to: {best_cfg_path}")
    print("Next: run a full (non-Optuna) training pass with this config against the complete dataset.")


if __name__ == "__main__":
    main()
