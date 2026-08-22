"""3D Live Viewer for sandbox environment.

Visualizes panel orientation, weather conditions, and PINN predictions
in real-time during RL training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class Viewer3D:
    """3D viewer for sandbox environment visualization.

    Currently a logging/metrics viewer. Full 3D visualization would require:
    - PyOpenGL or similar for 3D rendering
    - Real-time update mechanisms
    - Event handling for interaction

    For now, we provide:
    - Matplotlib-based plots
    - Text-based visualization
    - Metrics logging
    """

    def __init__(
        self,
        output_dir: Path | str = "outputs/sandbox/viewer",
        enabled: bool = True,
        fps: int = 30,
    ) -> None:
        """Initialize viewer.

        Args:
            output_dir: Directory for saving plots/logs
            enabled: Whether viewer is active
            fps: Target frame rate (for real-time mode)
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.fps = fps
        self.frame_count = 0

        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def render_state(
        self,
        pose: dict[str, float],
        weather: dict[str, float],
        pinn_pred: dict[str, float],
        rk4_pred: Optional[dict[str, float]] = None,
        reward_breakdown: Optional[dict[str, float | str]] = None,
        discrepancy: Optional[dict[str, float]] = None,
        decision_reason: str = "",
        episode: int = 0,
        step: int = 0,
    ) -> None:
        """Render current environment state.

        Args:
            pose: Panel pose {pitch, yaw, roll, z}
            weather: Weather conditions {T_amb, wind, irradiance, clouds, ...}
            pinn_pred: PINN predictions {T_operating, eta, ...}
            rk4_pred: Optional RK4TRAN ground truth
            episode: Episode number
            step: Step within episode
        """
        if not self.enabled:
            return

        self.frame_count += 1

        # Text-based visualization (can be enhanced with matplotlib later)
        state_str = self._format_state(
            pose,
            weather,
            pinn_pred,
            rk4_pred,
            reward_breakdown,
            discrepancy,
            decision_reason,
        )
        self._log_state(state_str, episode, step)

    def _format_state(
        self,
        pose: dict[str, float],
        weather: dict[str, float],
        pinn_pred: dict[str, float],
        rk4_pred: Optional[dict[str, float]] = None,
        reward_breakdown: Optional[dict[str, float | str]] = None,
        discrepancy: Optional[dict[str, float]] = None,
        decision_reason: str = "",
    ) -> str:
        """Format state for display.

        Args:
            pose: Panel pose
            weather: Weather
            pinn_pred: PINN predictions
            rk4_pred: RK4TRAN predictions

        Returns:
            Formatted state string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("PANEL STATE")
        lines.append("-" * 70)
        lines.append(f"  Pitch: {pose.get('pitch', 0):.1f}°  | Yaw: {pose.get('yaw', 0):.1f}°")
        lines.append(f"  Roll:  {pose.get('roll', 0):.1f}°  | Height: {pose.get('z', 0):.2f}m")

        lines.append("WEATHER")
        lines.append("-" * 70)
        lines.append(f"  Temp: {weather.get('T_amb', 0):.1f}K | Wind: {weather.get('wind_speed', 0):.1f} m/s")
        lines.append(f"  Irrad: {weather.get('irradiance', 0):.0f} W/m² | Cloud: {weather.get('cloud_cover', 0):.1%}")

        lines.append("PREDICTIONS")
        lines.append("-" * 70)
        T_p = pinn_pred.get("T_operating", 0)
        eta_p = pinn_pred.get("eta", 0)
        lines.append(f"  PINN:  T = {T_p:.1f}K | η = {eta_p:.4f}")

        if rk4_pred:
            T_r = rk4_pred.get("T_operating", 0)
            eta_r = rk4_pred.get("eta", 0)
            if discrepancy:
                T_err = abs(discrepancy.get("T_operating", 0))
                eta_err = abs(discrepancy.get("eta", 0))
            else:
                T_err = abs(T_p - T_r)
                eta_err = abs(eta_p - eta_r)
            lines.append(f"  RK4:   T = {T_r:.1f}K | η = {eta_r:.4f}")
            lines.append(f"  Error: ΔT = {T_err:.2f}K | Δη = {eta_err:.6f}")

        if reward_breakdown:
            lines.append("REWARD")
            lines.append("-" * 70)
            lines.append(
                "  "
                f"capture={float(reward_breakdown.get('capture_reward', 0.0)):+.4f}  "
                f"temp={float(reward_breakdown.get('temp_penalty', 0.0)):+.4f}  "
                f"smooth={float(reward_breakdown.get('smoothness_penalty', 0.0)):+.4f}  "
                f"corr={float(reward_breakdown.get('correction_penalty', 0.0)):+.4f}"
            )
            lines.append(
                "  "
                f"origin={reward_breakdown.get('reward_origin', 'unknown')}  "
                f"total={float(reward_breakdown.get('total_reward', 0.0)):+.4f}"
            )

        if decision_reason:
            lines.append("DECISION")
            lines.append("-" * 70)
            lines.append(f"  {decision_reason}")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _log_state(self, state_str: str, episode: int, step: int) -> None:
        """Log state to file.

        Args:
            state_str: Formatted state string
            episode: Episode number
            step: Step number
        """
        log_file = self.output_dir / f"episode_{episode:04d}.txt"
        with open(log_file, "a") as f:
            f.write(f"[Step {step}]\n{state_str}\n\n")

    def plot_episode_summary(
        self,
        episode: int,
        rewards: list[float],
        T_errors: list[float],
        eta_errors: list[float],
    ) -> None:
        """Create episode summary plot.

        Args:
            episode: Episode number
            rewards: List of rewards per step
            T_errors: Temperature prediction errors
            eta_errors: Efficiency prediction errors
        """
        if not self.enabled:
            return

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Rewards
            axes[0].plot(rewards)
            axes[0].set_title("Episode Rewards")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Reward")
            axes[0].grid(True)

            # Temperature errors
            axes[1].plot(T_errors)
            axes[1].set_title("Temperature Prediction Error")
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("|T_PINN - T_RK4| (K)")
            axes[1].grid(True)

            # Efficiency errors
            axes[2].plot(eta_errors)
            axes[2].set_title("Efficiency Prediction Error")
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel("|η_PINN - η_RK4|")
            axes[2].grid(True)

            plot_file = self.output_dir / f"episode_{episode:04d}.png"
            plt.savefig(plot_file, dpi=100, bbox_inches="tight")
            plt.close()

        except ImportError:
            pass  # Matplotlib not available, skip plotting


class ViewerFactory:
    """Factory for creating viewers based on configuration."""

    @staticmethod
    def create_from_config(config: dict) -> Viewer3D:
        """Create viewer from config dict.

        Args:
            config: Viewer config (output_dir, enabled, fps, etc.)

        Returns:
            Configured Viewer3D instance
        """
        return Viewer3D(
            output_dir=config.get("output_dir", "outputs/sandbox/viewer"),
            enabled=config.get("enabled", True),
            fps=config.get("fps", 30),
        )
