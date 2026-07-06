"""Entry point for PINN fine-tuning after RL exploration."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    """Run fine-tuning pipeline."""
    parser = argparse.ArgumentParser(description="Fine-tune PINN after RL exploration")
    parser.add_argument("--config", type=str, default="configs/finetune.yaml", help="Config file path")
    parser.add_argument("--checkpoint", type=str, help="Override pretrained checkpoint path")
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
        config["paths"]["checkpoint_dir"] = args.checkpoint
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    print("Fine-tuning config:")
    print(yaml.dump(config))

    # TODO: Implement fine-tuning
    # 1. Load pre-trained PINN checkpoint
    # 2. Load RL experience data (if available)
    # 3. Create combined dataset (synthetic + RL experience)
    # 4. Run training loop with lower learning rate
    # 5. Save fine-tuned checkpoint

    print("Fine-tuning not yet implemented")
    print("TODO: Implement fine-tuning loop")


if __name__ == "__main__":
    main()
