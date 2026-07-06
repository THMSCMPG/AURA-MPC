"""Entry point for sandbox RL training and validation.

This script initializes the PINN with pre-trained weights and trains a control
policy in the sandbox environment with live 3D visualization.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Run sandbox RL training."""
    parser = argparse.ArgumentParser(description="Run sandbox RL training")
    parser.add_argument("--config", type=str, default="configs/sandbox.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, help="Override PINN checkpoint path")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Override config
    if args.checkpoint:
        config["model"]["checkpoint_path"] = args.checkpoint
    if args.epochs:
        config["sandbox"]["train_epochs"] = args.epochs

    print("Sandbox RL training config:")
    print(yaml.dump(config))

    # TODO: Implement sandbox training
    # 1. Load pre-trained PINN checkpoint
    # 2. Initialize policy network
    # 3. Initialize panel environment
    # 4. Initialize 3D viewer
    # 5. Run RL training loop:
    #    - Collect trajectories
    #    - Compare PINN vs RK4TRAN
    #    - Update policy
    #    - Visualize results
    # 6. Save policy checkpoint and metrics

    print("Sandbox RL training not yet implemented")
    print("TODO: Implement training loop")


if __name__ == "__main__":
    main()
