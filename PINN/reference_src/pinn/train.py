"""Minimal training scaffold for :class:`~src.pinn.model.PINNSurrogate`.

This module provides:

* :func:`train` — training loop with Adam + cosine-annealing, loss CSV
  logging, and best-checkpoint saving.
* :func:`main` — CLI entry-point that wires everything together with
  ``argparse``.

**DataLoader contract** — each batch must be a dict with the following keys:

* ``'G'`` — irradiance [W/m²], shape ``(B,)``
* ``'T_amb'`` — ambient temperature [°C], shape ``(B,)``
* ``'WS'`` — wind speed [m/s], shape ``(B,)``
* ``'tilt'`` — panel tilt [degrees], shape ``(B,)``
* ``'azimuth'`` — panel azimuth [degrees], shape ``(B,)``
* ``'height'`` — mounting height [m], shape ``(B,)``
* ``'T_target'`` — ground-truth panel temperature [°C], shape ``(B, N)``
* ``'P_target'`` — ground-truth max-power-point [W], shape ``(B, N)``
* ``'t'`` — time vector [s], shape ``(N,)`` (can be the same for every batch)

Usage (CLI)::

    python -m src.pinn.train \\
        --epochs 100 --lr 1e-3 \\
        --w-data 1.0 --w-phys 0.1 \\
        --run-dir runs/ --checkpoint-dir checkpoints/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .model import PINNSurrogate, log_law_ws_eff
from .physics_loss import pinn_loss

__all__ = ["train", "main"]


# Minimum LR as a fraction of the initial LR at the end of cosine annealing.
_MIN_LR_RATIO: float = 1e-3


# ---------------------------------------------------------------------------
# Training loop.
# ---------------------------------------------------------------------------


def _run_epoch(
    model: PINNSurrogate,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer | None,
    *,
    device: torch.device,
    w_data: float,
    w_phys: float,
) -> dict[str, float]:
    """Run one epoch (train or validation).

    When ``optimizer`` is ``None`` the model is evaluated in no-grad mode.
    """
    is_train = optimizer is not None
    model.train(is_train)
    totals: dict[str, float] = {
        "total": 0.0,
        "mse_data_T": 0.0,
        "mse_data_P": 0.0,
        "mse_phys": 0.0,
        "boundary": 0.0,
    }
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            G: Tensor = batch["G"].to(device)
            T_amb: Tensor = batch["T_amb"].to(device)
            WS: Tensor = batch["WS"].to(device)
            tilt: Tensor = batch["tilt"].to(device)
            azimuth: Tensor = batch["azimuth"].to(device)
            height: Tensor = batch["height"].to(device)
            T_target: Tensor = batch["T_target"].to(device)
            P_target: Tensor = batch["P_target"].to(device)
            t: Tensor = batch["t"].to(device)

            # Forward pass.
            T_pred, P_pred = model(G, T_amb, WS, tilt, azimuth, height)

            # Effective wind speed (same formula as the model's pre-processor).
            WS_eff = log_law_ws_eff(WS, height)

            loss_dict = pinn_loss(
                T_pred, P_pred, T_target, P_target, T_amb, t, G, WS_eff,
                w_data=w_data, w_phys=w_phys,
            )

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss_dict["total"].backward()  # type: ignore[no-untyped-call]
                optimizer.step()

            for k in totals:
                totals[k] += float(loss_dict[k].detach())
            n_batches += 1

    if n_batches > 0:
        for k in totals:
            totals[k] /= n_batches
    return totals


def train(
    model: PINNSurrogate,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    n_epochs: int = 100,
    lr: float = 1e-3,
    w_data: float = 1.0,
    w_phys: float = 0.1,
    run_dir: Path = Path("runs"),
    checkpoint_dir: Path = Path("checkpoints"),
    device: torch.device | str = "cpu",
) -> None:
    """Train a :class:`PINNSurrogate` with Adam + cosine-annealing LR.

    Training progress (all five loss components) is appended row-by-row to
    ``<run_dir>/loss_log.csv``.  The model checkpoint with the lowest
    validation ``mse_data_T`` is saved to ``<checkpoint_dir>/best.pt``.

    Args:
        model: The :class:`PINNSurrogate` to train (moved to ``device``
            in-place).
        train_loader: DataLoader yielding training batches (see module
            docstring for the required dict keys).
        val_loader: DataLoader yielding validation batches.
        n_epochs: Total number of training epochs.
        lr: Initial Adam learning rate.
        w_data: Weight for the data-fidelity MSE terms in the PINN loss.
        w_phys: Weight for the Fuentes ODE residual in the PINN loss.
        run_dir: Directory for the CSV loss log.  Created if absent.
        checkpoint_dir: Directory for ``best.pt``.  Created if absent.
        device: Torch device to train on (default CPU).
    """
    _device = torch.device(device)
    model.to(_device)

    run_dir = Path(run_dir)
    checkpoint_dir = Path(checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * _MIN_LR_RATIO)

    csv_path = run_dir / "loss_log.csv"
    fieldnames = ["epoch", "total", "mse_data_T", "mse_data_P", "mse_phys", "boundary",
                  "val_total", "val_mse_data_T", "val_mse_data_P", "val_mse_phys",
                  "val_boundary"]

    best_val_mse_T = math.inf

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, n_epochs + 1):
            train_metrics = _run_epoch(
                model, train_loader, optimizer,
                device=_device, w_data=w_data, w_phys=w_phys,
            )
            val_metrics = _run_epoch(
                model, val_loader, None,
                device=_device, w_data=w_data, w_phys=w_phys,
            )

            scheduler.step()

            row: dict[str, Any] = {"epoch": epoch}
            for k, v in train_metrics.items():
                row[k] = v
            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            writer.writerow(row)
            fh.flush()

            # Save best checkpoint by validation mse_data_T.
            if val_metrics["mse_data_T"] < best_val_mse_T:
                best_val_mse_T = val_metrics["mse_data_T"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_mse_data_T": best_val_mse_T,
                        "w_data": w_data,
                        "w_phys": w_phys,
                    },
                    checkpoint_dir / "best.pt",
                )


# ---------------------------------------------------------------------------
# CLI entry-point.
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinn-surrogate-train",
        description="Train the PINNSurrogate model.",
    )
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial Adam learning rate (default: 1e-3).")
    p.add_argument("--w-data", type=float, default=1.0,
                   help="Weight for data-fidelity loss terms (default: 1.0).")
    p.add_argument("--w-phys", type=float, default=0.1,
                   help="Weight for physics (ODE residual) loss term (default: 0.1).")
    p.add_argument("--n-layers", type=int, default=6,
                   help="Number of hidden layers in PINNSurrogate (default: 6).")
    p.add_argument("--hidden-dim", type=int, default=128,
                   help="Hidden layer width (default: 128).")
    p.add_argument("--n-timesteps", type=int, default=96,
                   help="Number of output timesteps (default: 96).")
    p.add_argument("--run-dir", type=Path, default=Path("runs"),
                   help="Directory for loss CSV log (default: runs/).")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"),
                   help="Directory for best.pt checkpoint (default: checkpoints/).")
    p.add_argument("--device", type=str, default="cpu",
                   help="Torch device string (default: cpu).")
    return p


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI entry-point for surrogate training.

    This function expects ``train_loader`` and ``val_loader`` to be provided
    externally (e.g. by an integration script that constructs the dataset).
    In standalone mode it creates small random datasets for smoke-testing.
    """
    args = _build_parser().parse_args(argv)

    # ---- build a tiny synthetic dataset for smoke-testing ---------------

    torch.manual_seed(0)
    B, N = 32, args.n_timesteps
    t_vec = torch.linspace(0.0, 86_400.0, N)

    def _make_batch(n: int) -> dict[str, Tensor]:
        return {
            "G": torch.rand(n) * 1000.0,
            "T_amb": torch.rand(n) * 30.0 + 10.0,
            "WS": torch.rand(n) * 10.0,
            "tilt": torch.rand(n) * 45.0,
            "azimuth": (torch.rand(n) - 0.5) * 360.0,
            "height": torch.rand(n) * 3.0 + 0.5,
            "T_target": torch.rand(n, N) * 40.0 + 15.0,
            "P_target": torch.rand(n, N) * 300.0,
            "t": t_vec,
        }

    # Wrap in a simple list-of-dicts DataLoader.
    train_data = [_make_batch(B) for _ in range(4)]
    val_data = [_make_batch(B) for _ in range(2)]

    model = PINNSurrogate(
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        n_timesteps=args.n_timesteps,
    )

    train(
        model,
        train_data,  # type: ignore[arg-type]
        val_data,    # type: ignore[arg-type]
        n_epochs=args.epochs,
        lr=args.lr,
        w_data=args.w_data,
        w_phys=args.w_phys,
        run_dir=args.run_dir,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
