"""Entry point for sandbox RL training with PINN validation.

This script initializes the pre-trained PINN and trains a control policy
in the sandbox environment with RK4TRAN validation and live visualization.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sandbox import train_sandbox_policy


def main() -> None:
    """Run sandbox RL training."""
    parser = argparse.ArgumentParser(description="Train control policy in sandbox environment")
    parser.add_argument("--config", type=str, default="configs/sandbox.yaml", help="Config file path")
    parser.add_argument("--pinn-checkpoint", type=str, help="Override PINN checkpoint path")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--episodes-per-epoch", type=int, help="Override episodes per epoch")
    parser.add_argument("--output-dir", type=str, default="outputs/sandbox", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Device setup
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config["device"] = device
    print(f"Using device: {device}")

    # Override config with CLI args
    if args.pinn_checkpoint:
        config["model"]["checkpoint_path"] = args.pinn_checkpoint
    if args.epochs:
        config["sandbox"]["train_epochs"] = args.epochs
    if args.episodes_per_epoch:
        config["sandbox"]["episodes_per_epoch"] = args.episodes_per_epoch

    # Get PINN checkpoint
    pinn_checkpoint = config.get("model", {}).get("checkpoint_path")
    if not pinn_checkpoint:
        raise ValueError("PINN checkpoint path not specified in config or CLI args")

    pinn_checkpoint = Path(pinn_checkpoint)
    if not pinn_checkpoint.exists():
        print(f"Warning: PINN checkpoint not found: {pinn_checkpoint}")
        print("  This is expected for the first run.")
        print("  You should run pre-training first: python scripts/run_pretrain.py")

    print("\nSandbox RL training config:")
    print(yaml.dump(config, default_flow_style=False))

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    print("\nStarting sandbox RL training...")
    try:
        artifacts = train_sandbox_policy(
            pinn_checkpoint=pinn_checkpoint,
            config=config,
            output_dir=output_dir,
        )

        print(f"\n✓ Training complete!")
        print(f"  Policy checkpoint: {artifacts.policy_checkpoint}")
        print(f"  Metrics file: {artifacts.metrics_file}")
        print(f"  Log directory: {artifacts.log_dir}")

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
