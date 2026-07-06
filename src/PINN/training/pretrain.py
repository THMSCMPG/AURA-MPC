"""Pre-training loop for PINN on RK4TRAN synthetic data."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.uncertainty import UncertaintyProcessor
from models import PINNSurrogate


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute weighted MSE loss.

    Args:
        pred: Predictions [B, ...]
        target: Ground truth [B, ...]
        weights: Optional loss weights [B]

    Returns:
        Scalar loss
    """
    mse = F.mse_loss(pred, target, reduction="none")
    if weights is not None:
        mse = mse * weights.unsqueeze(-1)
    return mse.mean()


def compute_loss(
    batch: dict,
    model_output: dict,
    uq_processor: Optional[UncertaintyProcessor] = None,
) -> dict[str, torch.Tensor]:
    """Compute training loss with optional UQ weighting.

    Args:
        batch: Input batch dict
        model_output: Model predictions dict
        uq_processor: Optional uncertainty processor

    Returns:
        Dict with loss components
    """
    losses = {}

    # Temperature loss
    T_loss = weighted_mse_loss(
        model_output["T_operating"],
        batch["T_operating"].unsqueeze(-1),
    )
    losses["T_loss"] = T_loss

    # Efficiency loss
    eta_loss = weighted_mse_loss(
        model_output["eta"],
        batch["eta"].unsqueeze(-1),
    )
    losses["eta_loss"] = eta_loss

    # Combine losses
    losses["total"] = T_loss + eta_loss

    return losses


class Trainer:
    """PINN pre-trainer on RK4TRAN synthetic data."""

    def __init__(
        self,
        model: PINNSurrogate,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize trainer.

        Args:
            model: PINN model to train
            device: Device to train on
            lr: Learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.uq_processor = UncertaintyProcessor(strategy="weighted")

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Run single training epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dict with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            # Forward pass
            x = torch.cat(
                [batch["weather"], batch["panel_state"], batch["location"]],
                dim=1,
            )
            if batch["time"].shape[-1] > 0:
                x = torch.cat([x, batch["time"]], dim=1)

            model_output = self.model(x)

            # Compute loss
            losses = compute_loss(batch, model_output, self.uq_processor)
            loss = losses["total"]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Run validation epoch.

        Args:
            val_loader: Validation data loader

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            x = torch.cat(
                [batch["weather"], batch["panel_state"], batch["location"]],
                dim=1,
            )
            if batch["time"].shape[-1] > 0:
                x = torch.cat([x, batch["time"]], dim=1)

            model_output = self.model(x)
            losses = compute_loss(batch, model_output, self.uq_processor)
            loss = losses["total"]

            total_loss += loss.item()
            num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs

        Returns:
            Dict with training history
        """
        checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        log_dir = Path(log_dir or "logs")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        writer = SummaryWriter(str(log_dir / "pretrain"))

        # Cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=1e-6,
        )

        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val_loss = float("inf")
        best_checkpoint = checkpoint_dir / "best_model.pt"

        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.scheduler_step = scheduler.step

            # Validate
            val_metrics = self.validate(val_loader)

            # Log
            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["lr"].append(current_lr)

            writer.add_scalar("loss/train", train_metrics["train_loss"], epoch)
            writer.add_scalar("loss/val", val_metrics["val_loss"], epoch)
            writer.add_scalar("learning_rate", current_lr, epoch)

            # Checkpoint best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                torch.save(self.model.state_dict(), best_checkpoint)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"train_loss={train_metrics['train_loss']:.6f}, "
                    f"val_loss={val_metrics['val_loss']:.6f}, "
                    f"lr={current_lr:.2e}"
                )

            scheduler.step()

        writer.close()

        # Save final model
        torch.save(self.model.state_dict(), checkpoint_dir / "final_model.pt")

        # Save history to CSV
        metrics_file = log_dir / "metrics.csv"
        with open(metrics_file, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
            writer_csv.writeheader()
            for epoch, (tl, vl, lr) in enumerate(zip(history["train_loss"], history["val_loss"], history["lr"])):
                writer_csv.writerow({"epoch": epoch, "train_loss": tl, "val_loss": vl, "lr": lr})

        print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
        print(f"Checkpoints saved to {checkpoint_dir}")
        print(f"Metrics saved to {metrics_file}")

        return history
