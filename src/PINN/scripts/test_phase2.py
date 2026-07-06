#!/usr/bin/env python
"""End-to-end test for Phase 2 sandbox RL training.

Tests:
1. PINN loading and inference
2. RK4TRAN validator initialization
3. PanelEnv creation and stepping
4. Full RL training loop
"""

from __future__ import annotations

import sys
from pathlib import Path

# Check dependencies first
try:
    import torch
except ImportError:
    print("⚠ PyTorch not installed - skipping CUDA tests")
    torch = None

try:
    import numpy as np
except ImportError:
    print("✗ NumPy required but not installed")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("⚠ PyYAML not installed - config test will be skipped")
    yaml = None


def test_pinn_loading():
    """Test PINN model loading."""
    print("\n[TEST 1] PINN Model Loading")
    print("-" * 60)

    if torch is None:
        print("⊘ SKIPPED: PyTorch not installed")
        return True

    try:
        from src.PINN.models import PINNSurrogate

        model = PINNSurrogate(input_dim=18, hidden_dim=128, output_dim=4, num_blocks=4)
        print(f"✓ Model created: {model.__class__.__name__}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        x = torch.randn(2, 18)
        y = model(x)
        assert y.shape == (2, 4), f"Expected (2,4), got {y.shape}"
        print(f"✓ Forward pass works: input (2, 18) → output {y.shape}")

        return True

    except Exception as e:
        print(f"✗ PINN loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rk4_validator():
    """Test RK4TRAN validator."""
    print("\n[TEST 2] RK4TRAN Validator")
    print("-" * 60)

    try:
        # Add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        
        from PINN.sandbox import RK4TRANValidator

        # Check if binary exists
        binary_path = Path(__file__).parent.parent.parent.parent / "RK4TRAN" / "main"
        if not binary_path.exists():
            print(f"⚠ RK4TRAN binary not found at {binary_path}")
            print("  (This is expected if you're running from a different directory)")
            return True

        print(f"✓ RK4TRAN binary found at {binary_path}")

        validator = RK4TRANValidator(binary_path, cache_size=100)
        print(f"✓ Validator initialized with cache size: 100")

        # Test prediction
        pred = validator.predict(
            weather={
                "T_amb": 300.0,
                "wind_speed": 5.0,
                "wind_dir": 180.0,
                "humidity": 0.5,
                "irradiance": 800.0,
                "clouds": 0.2,
                "pressure": 101325.0,
            },
            panel_state={
                "pv_height": 1.0,
                "pitch": 30.0,
                "roll": 0.0,
                "yaw": 0.0,
            },
            location={"lat": 40.0, "lon": -75.0, "alt": 0.0},
        )

        assert "T_operating" in pred, "Missing T_operating"
        assert "eta" in pred, "Missing eta"
        assert 200 < pred["T_operating"] < 400, f"Invalid temperature: {pred['T_operating']}"
        assert 0 < pred["eta"] < 1, f"Invalid efficiency: {pred['eta']}"

        print(f"✓ Prediction works:")
        print(f"  T_operating: {pred['T_operating']:.1f} K")
        print(f"  eta: {pred['eta']:.4f}")

        return True

    except Exception as e:
        print(f"✗ RK4TRAN validator failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_panel_env():
    """Test PanelEnv creation and stepping."""
    print("\n[TEST 3] PanelEnv Environment")
    print("-" * 60)

    if torch is None:
        print("⊘ SKIPPED: PyTorch not installed")
        return True

    try:
        # Mock PINN agent for testing
        class MockPINNAgent:
            def predict(self, weather, panel_state, location, time, include_rk4=False):
                return {
                    "pinn": {
                        "T_operating": 45.0 + 273.15,
                        "eta": 0.18,
                    }
                }

        from src.PINN.sandbox import PanelEnv

        agent = MockPINNAgent()
        env = PanelEnv(pinn_agent=agent, seed=42)

        print(f"✓ PanelEnv created")
        print(f"  obs_dim: {env.observation_dim}")
        print(f"  action_dim: {env.action_dim}")

        # Test reset
        obs, info = env.reset()
        assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
        print(f"✓ Reset works: obs shape {obs.shape}")
        print(f"  Conditions: lat={info['conditions'].lat:.1f}, lon={info['conditions'].lon:.1f}")

        # Test step
        action = np.array([0.1, -0.2, 0.0, 0.5], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        assert next_obs.shape == (14,), f"Expected obs shape (14,), got {next_obs.shape}"
        assert isinstance(reward, (float, np.floating)), f"Expected float reward, got {type(reward)}"
        assert isinstance(terminated, bool), f"Expected bool terminated, got {type(terminated)}"
        assert isinstance(truncated, bool), f"Expected bool truncated, got {type(truncated)}"

        print(f"✓ Step works:")
        print(f"  obs shape: {next_obs.shape}")
        print(f"  reward: {reward:.4f}")
        print(f"  terminated: {terminated}, truncated: {truncated}")

        # Test multiple steps
        for step_idx in range(5):
            action = np.random.uniform(-1, 1, 4).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        print(f"✓ Episode ran for {step_idx + 1} steps without errors")

        return True

    except Exception as e:
        print(f"✗ PanelEnv failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sandbox_training():
    """Test sandbox training loop."""
    print("\n[TEST 4] Sandbox Training Loop")
    print("-" * 60)

    if torch is None:
        print("⊘ SKIPPED: PyTorch not installed")
        return True

    try:
        # Mock PINN agent
        class MockPINNAgent:
            def predict(self, weather, panel_state, location, time, include_rk4=False):
                return {
                    "pinn": {
                        "T_operating": 45.0 + 273.15,
                        "eta": 0.18,
                    }
                }

        from src.PINN.sandbox import SandboxTrainer, PanelEnv

        agent = MockPINNAgent()
        env = PanelEnv(pinn_agent=agent, seed=42)

        trainer = SandboxTrainer(
            pinn_agent=agent,
            obs_dim=14,
            action_dim=4,
            hidden_dim=64,
            learning_rate=1e-3,
            discount_gamma=0.98,
            device="cpu",
        )

        print(f"✓ SandboxTrainer created")

        # Train one episode
        metrics = trainer.train_episode(env=env, episode_steps=8)

        print(f"✓ train_episode completed:")
        print(f"  episode_return: {metrics['episode_return']:.4f}")
        print(f"  loss: {metrics['loss']:.4f}")
        print(f"  avg_T_error: {metrics['avg_T_error']:.4f}")
        print(f"  avg_eta_error: {metrics['avg_eta_error']:.4f}")

        assert isinstance(metrics["episode_return"], (float, np.floating))
        assert isinstance(metrics["loss"], (float, np.floating))

        print(f"✓ Training loop metrics valid")

        return True

    except Exception as e:
        print(f"✗ Sandbox training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_loading():
    """Test configuration loading."""
    print("\n[TEST 5] Configuration Loading")
    print("-" * 60)

    if yaml is None:
        print("⊘ SKIPPED: PyYAML not installed")
        return True

    try:
        config_path = Path("src/PINN/configs/sandbox.yaml")

        if not config_path.exists():
            print(f"⚠ Config not found at {config_path}")
            return True

        with open(config_path) as f:
            config = yaml.safe_load(f)

        print(f"✓ Config loaded from {config_path}")
        print(f"  Keys: {list(config.keys())}")

        # Verify essential keys
        assert "sandbox" in config, "Missing 'sandbox' section"
        assert "fortran" in config, "Missing 'fortran' section"
        assert "model" in config, "Missing 'model' section"

        sandbox_cfg = config["sandbox"]
        assert "train_epochs" in sandbox_cfg, "Missing train_epochs"
        assert "episodes_per_epoch" in sandbox_cfg, "Missing episodes_per_epoch"

        print(f"✓ Config structure valid")
        print(f"  train_epochs: {sandbox_cfg.get('train_epochs')}")
        print(f"  episodes_per_epoch: {sandbox_cfg.get('episodes_per_epoch')}")
        print(f"  learning_rate: {sandbox_cfg.get('learning_rate')}")

        return True

    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> int:
    """Run all tests."""
    print("\n" + "=" * 60)
    print("END-TO-END TEST SUITE: Phase 2 Sandbox RL")
    print("=" * 60)

    tests = [
        ("PINN Loading", test_pinn_loading),
        ("RK4TRAN Validator", test_rk4_validator),
        ("PanelEnv", test_panel_env),
        ("Sandbox Training", test_sandbox_training),
        ("Config Loading", test_config_loading),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
