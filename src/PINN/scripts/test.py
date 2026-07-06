"""Quick test of data loading and model forward pass."""

import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import RK4TRANDataset, create_dataloaders
from models import PINNSurrogate


def test_data_loading(csv_dir: Path, num_samples: int = 5) -> None:
    """Test RK4TRAN data loading."""
    print(f"Testing data loading from {csv_dir}")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_dir}")
        return

    print(f"Found {len(csv_files)} CSV files")

    try:
        dataset = RK4TRANDataset(csv_files)
        print(f"✓ Loaded {len(dataset)} samples")

        # Test sampling
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            print(f"\nSample {i}:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_dataloaders(csv_dir: Path, batch_size: int = 4) -> None:
    """Test DataLoader creation."""
    print(f"\nTesting DataLoader creation")

    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found in {csv_dir}")
        return

    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            csv_files,
            batch_size=batch_size,
            train_split=0.8,
            val_split=0.1,
            num_workers=0,
        )
        print(f"✓ Created dataloaders")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        # Sample one batch
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape {value.shape}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def test_model() -> None:
    """Test PINN model."""
    print(f"\nTesting PINN model")

    try:
        model = PINNSurrogate(
            input_dim=18,
            hidden_dim=128,
            num_residual_blocks=4,
        )
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Test forward pass
        x = torch.randn(4, 18)  # Batch of 4, 18 features
        output = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Output keys: {list(output.keys())}")
        for key, value in output.items():
            print(f"  {key}: shape {value.shape}")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """Run all tests."""
    print("=" * 60)
    print("PINN Pipeline Test Suite")
    print("=" * 60)

    # Test 1: Model
    test_model()

    # Test 2: Data loading (if CSV files exist)
    csv_dir = Path(__file__).parent.parent.parent / "RK4TRAN" / "work"
    if csv_dir.exists():
        test_data_loading(csv_dir, num_samples=3)
        test_dataloaders(csv_dir, batch_size=4)
    else:
        print(f"\nSkipping data loading tests (CSV dir not found: {csv_dir})")
        print("To test with real data, ensure RK4TRAN CSV files are available")

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
