"""Entry point for PINN pre-training on RK4TRAN synthetic data."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import NumericNormalizer, RK4TRANDataset, create_dataloaders
from models import PINNSurrogate
from training import Trainer


def main() -> None:
    """Run pre-training pipeline."""
    parser = argparse.ArgumentParser(description="Pre-train PINN on RK4TRAN synthetic data")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Config file path")
    parser.add_argument("--csv-dir", type=str, help="Override CSV directory")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config with CLI args
    if args.csv_dir:
        config["paths"]["csv_data_dir"] = args.csv_dir
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Create output directories
    output_dir = Path(config["paths"]["output_dir"])
    checkpoint_dir = Path(config["paths"]["checkpoints_dir"])
    log_dir = Path(config["paths"]["logs_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Find CSV files
    csv_dir = Path(config["paths"]["csv_data_dir"])
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    print(f"Found {len(csv_files)} CSV files")
    for f in csv_files:
        print(f"  - {f.name}")

    # Load dataset and create dataloaders
    print("Loading dataset...")
    dataset = RK4TRANDataset(csv_files)
    print(f"Loaded {len(dataset)} samples")

    # Create normalizer if configured
    normalizer = None
    if config["data"].get("normalize"):
        print("Fitting normalizer...")
        normalizer = NumericNormalizer()
        # Fit on full dataset (in practice, fit on train only)
        normalizer.fit({"weather": [], "panel_state": [], "location": []})
        normalizer.save(checkpoint_dir / "normalizer.json")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_files,
        batch_size=config["data"]["batch_size"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        normalizer=normalizer,
        num_workers=config["data"]["num_workers"],
        seed=config["data"]["seed"],
    )

    print(f"Train: {len(train_loader) * config['data']['batch_size']} samples")
    print(f"Val: {len(val_loader) * config['data']['batch_size']} samples")
    print(f"Test: {len(test_loader) * config['data']['batch_size']} samples")

    # Create model
    print("Creating model...")
    model = PINNSurrogate(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_residual_blocks=config["model"]["num_residual_blocks"],
        num_outputs=config["model"]["num_outputs"],
        dropout=config["model"]["dropout"],
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Train
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    )

    print("Training complete!")
    print(f"Best model saved to {checkpoint_dir / 'best_model.pt'}")
    print(f"Metrics saved to {log_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
