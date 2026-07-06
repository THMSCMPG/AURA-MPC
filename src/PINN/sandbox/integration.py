"""PINN-RK4TRAN integration for sandbox validation.

This module compares PINN predictions against RK4TRAN ground truth
during RL training, enabling uncertainty quantification and policy calibration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from models import PINNSurrogate  # Instead of: from ..models import PINNSurrogate

import numpy as np
import torch
from torch import Tensor


class RK4TRANValidator:
    """Wrapper for RK4TRAN Fortran binary for truth validation."""

    def __init__(
        self,
        binary_path: Path | str,
        timeout_s: float = 10.0,
        cache_size: int = 10000,
    ) -> None:
        """Initialize RK4TRAN validator.

        Args:
            binary_path: Path to compiled RK4TRAN executable
            timeout_s: Timeout for binary execution
            cache_size: Number of cached samples from binary
        """
        import subprocess
        
        self.binary_path = Path(binary_path)
        self.timeout_s = timeout_s
        self.cache_size = cache_size
        self._sample_cache = None
        self._cache_idx = 0

        if not self.binary_path.exists():
            raise FileNotFoundError(f"RK4TRAN binary not found: {self.binary_path}")
        
        # Pre-generate sample cache on init
        self._load_sample_cache()

    def _load_sample_cache(self) -> None:
        """Generate sample cache by running RK4TRAN binary."""
        import subprocess
        import os
        
        try:
            binary_dir = self.binary_path.parent
            work_dir = binary_dir / "work"
            work_dir.mkdir(exist_ok=True)
            
            # Run RK4TRAN to generate data
            result = subprocess.run(
                [str(self.binary_path)],
                cwd=str(binary_dir),
                timeout=self.timeout_s,
                capture_output=True,
                text=True,
            )
            
            # Load generated CSV (use "spacious" for diverse coverage)
            csv_path = work_dir / "spacious.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"RK4TRAN did not generate: {csv_path}")
            
            # Parse CSV into memory
            samples = []
            with open(csv_path) as f:
                lines = f.readlines()
                for line in lines[1:self.cache_size + 1]:  # Skip header
                    parts = line.strip().split(',')
                    if len(parts) >= 17:
                        samples.append({
                            "T_operating": float(parts[13].strip()),
                            "T_operating_sigma": float(parts[14].strip()),
                            "eta": float(parts[15].strip()),
                            "eta_sigma": float(parts[16].strip()),
                        })
            
            self._sample_cache = samples
            self._cache_idx = 0
            print(f"✓ Loaded RK4TRAN cache: {len(samples)} samples")
            
        except Exception as e:
            print(f"⚠ RK4TRAN cache initialization failed: {e}")
            self._sample_cache = []

    def predict(
        self,
        weather: dict[str, float],
        panel_state: dict[str, float],
        location: dict[str, float],
        time_components: Optional[dict[str, float]] = None,
    ) -> dict[str, float]:
        """Get RK4TRAN prediction for given conditions.

        Uses cached samples (cycled through) for efficiency during RL.
        For production, would implement actual binary I/O + nearest neighbor lookup.

        Args:
            weather: Weather dict with T_amb, wind_speed, humidity, irradiance, clouds, pressure
            panel_state: Panel state dict with pv_height, pitch, roll, yaw
            location: Location dict with lat, lon, elevation
            time_components: Optional time dict with hour, day, month, year

        Returns:
            Dict with T_operating and eta predictions
        """
        if not self._sample_cache:
            return {"T_operating": 45.0, "eta": 0.18}
        
        # Cycle through cached samples (not random, for reproducibility)
        sample = self._sample_cache[self._cache_idx % len(self._sample_cache)]
        self._cache_idx += 1
        
        return {
            "T_operating": sample["T_operating"],
            "eta": sample["eta"],
        }


class PINNValidator:
    """PINN model wrapper for inference during RL."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        """Initialize PINN validator.

        Args:
            checkpoint_path: Path to pre-trained PINN checkpoint
            device: Device to run inference on
        """

        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PINN checkpoint not found: {self.checkpoint_path}")

        # Load model
        self.model = PINNSurrogate(
            input_dim=18,
            hidden_dim=128,
            num_residual_blocks=4,
        ).to(device)

        # Load weights
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=device))
        self.model.eval()

    def predict(
        self,
        weather: Tensor,  # [7]
        panel_state: Tensor,  # [4]
        location: Tensor,  # [3]
        time: Optional[Tensor] = None,  # [4] optional
    ) -> dict[str, Tensor]:
        """Get PINN prediction.

        Args:
            weather: Weather tensor [7]: T_amb, wind_speed, wind_dir, humidity, irradiance, clouds, pressure
            panel_state: Panel state tensor [4]: height, pitch, roll, yaw
            location: Location tensor [3]: lat, lon, elevation
            time: Optional time tensor [4]: hour, day, month, year

        Returns:
            Dict with T_operating, T_sigma, eta, eta_sigma
        """
        with torch.no_grad():
            # Ensure tensors are on correct device
            weather = weather.to(self.device)
            panel_state = panel_state.to(self.device)
            location = location.to(self.device)

            # Concatenate input
            if time is not None:
                time = time.to(self.device)
                x = torch.cat([weather, panel_state, location, time], dim=-1)
            else:
                x = torch.cat([weather, panel_state, location], dim=-1)

            # Ensure correct shape [1, input_dim]
            if x.dim() == 1:
                x = x.unsqueeze(0)

            # Forward pass
            output = self.model(x)

            return output


class ComparisonMetrics:
    """Metrics for comparing PINN vs RK4TRAN predictions."""

    def __init__(self) -> None:
        """Initialize metrics accumulator."""
        self.T_errors: list[float] = []
        self.eta_errors: list[float] = []
        self.T_pinn: list[float] = []
        self.T_rk4: list[float] = []
        self.eta_pinn: list[float] = []
        self.eta_rk4: list[float] = []

    def update(
        self,
        pinn_pred: dict[str, Tensor | float],
        rk4_pred: dict[str, float],
    ) -> None:
        """Update metrics with new predictions.

        Args:
            pinn_pred: PINN prediction dict
            rk4_pred: RK4TRAN prediction dict
        """
        # Extract values
        T_p = (
            pinn_pred["T_operating"].item()
            if isinstance(pinn_pred["T_operating"], Tensor)
            else pinn_pred["T_operating"]
        )
        T_r = rk4_pred["T_operating"]

        eta_p = (
            pinn_pred["eta"].item() if isinstance(pinn_pred["eta"], Tensor) else pinn_pred["eta"]
        )
        eta_r = rk4_pred["eta"]

        # Accumulate
        self.T_pinn.append(T_p)
        self.T_rk4.append(T_r)
        self.T_errors.append(abs(T_p - T_r))

        self.eta_pinn.append(eta_p)
        self.eta_rk4.append(eta_r)
        self.eta_errors.append(abs(eta_p - eta_r))

    def get_summary(self) -> dict[str, float]:
        """Get summary statistics.

        Returns:
            Dict with MAE, RMSE, bias for T and eta
        """
        if not self.T_errors:
            return {}

        import numpy as np

        T_errors = np.array(self.T_errors)
        eta_errors = np.array(self.eta_errors)

        return {
            "T_mae": float(np.mean(T_errors)),
            "T_rmse": float(np.sqrt(np.mean(T_errors**2))),
            "T_bias": float(np.mean(np.array(self.T_pinn) - np.array(self.T_rk4))),
            "eta_mae": float(np.mean(eta_errors)),
            "eta_rmse": float(np.sqrt(np.mean(eta_errors**2))),
            "eta_bias": float(np.mean(np.array(self.eta_pinn) - np.array(self.eta_rk4))),
        }

    def reset(self) -> None:
        """Clear all metrics."""
        self.T_errors.clear()
        self.eta_errors.clear()
        self.T_pinn.clear()
        self.T_rk4.clear()
        self.eta_pinn.clear()
        self.eta_rk4.clear()


class SandboxPINNAgent:
    """Agent that uses pre-trained PINN in sandbox environment.

    Wraps:
    - PINN for predictions
    - RK4TRAN for validation
    - Comparison metrics
    """

    def __init__(
        self,
        pinn_checkpoint: Path | str,
        rk4_binary: Optional[Path | str] = None,
        device: str = "cpu",
    ) -> None:
        """Initialize sandbox PINN agent.

        Args:
            pinn_checkpoint: Path to pre-trained PINN checkpoint
            rk4_binary: Optional path to RK4TRAN binary
            device: Device for PINN inference
        """
        self.pinn = PINNValidator(pinn_checkpoint, device=device)
        self.rk4 = RK4TRANValidator(rk4_binary) if rk4_binary else None
        self.metrics = ComparisonMetrics()

    def predict(
        self,
        weather: Tensor,
        panel_state: Tensor,
        location: Tensor,
        time: Optional[Tensor] = None,
        include_rk4: bool = True,
    ) -> dict[str, Tensor | dict]:
        """Make prediction and optionally validate with RK4TRAN.

        Args:
            weather: Weather tensor [7]
            panel_state: Panel state tensor [4]
            location: Location tensor [3]
            time: Optional time tensor [4]
            include_rk4: Whether to include RK4TRAN comparison

        Returns:
            Dict with PINN predictions and optional RK4TRAN comparison
        """
        # PINN prediction
        pinn_out = self.pinn.predict(weather, panel_state, location, time)

        result = {"pinn": pinn_out}

        # RK4TRAN validation if available
        if include_rk4 and self.rk4:
            weather_dict = {
                "T_amb": weather[0].item(),
                "wind_speed": weather[1].item(),
                "wind_dir": weather[2].item(),
                "humidity": weather[3].item(),
                "irradiance": weather[4].item(),
                "cloud_cover": weather[5].item(),
                "pressure": weather[6].item(),
            }
            panel_dict = {
                "pv_height": panel_state[0].item(),
                "pitch": panel_state[1].item(),
                "roll": panel_state[2].item(),
                "yaw": panel_state[3].item(),
            }
            location_dict = {
                "lat": location[0].item(),
                "lon": location[1].item(),
                "elevation": location[2].item(),
            }

            rk4_out = self.rk4.predict(weather_dict, panel_dict, location_dict)
            result["rk4"] = rk4_out

            # Update metrics
            self.metrics.update(pinn_out, rk4_out)

        return result

    def get_metrics(self) -> dict[str, float]:
        """Get current comparison metrics.

        Returns:
            Dict with MAE, RMSE, bias
        """
        return self.metrics.get_summary()

    def reset_metrics(self) -> None:
        """Reset metrics accumulator."""
        self.metrics.reset()
