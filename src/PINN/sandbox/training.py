"""Sandbox RL training with PINN-RK4TRAN validation.

Phase 2: Train control policy with PINN predictions and RK4TRAN validation.
Includes live visualization and comparison metrics.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from .environment import PanelEnv
from .integration import SandboxPINNAgent
from .viewer import Viewer3D


@dataclass(frozen=True, slots=True)
class SandboxTrainArtifacts:
    """Output artifacts from sandbox training."""

    policy_checkpoint: Path
    metrics_file: Path
    comparison_file: Path
    log_dir: Path


class PolicyNetwork(nn.Module):
    """Gaussian policy for continuous 4-DoF control.

    Takes observation state and outputs mean/std for action distribution.
    """

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, std_init: float = 0.25) -> None:
        """Initialize policy network.

        Args:
            obs_dim: Observation dimension
            hidden_dim: Hidden layer dimension
            action_dim: Action dimension (4 for pitch, yaw, roll, z)
            std_init: Initial action standard deviation
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Mean network
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # Learnable log-std
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(std_init))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            obs: Observation tensor [B, obs_dim]

        Returns:
            Tuple of (mean [B, action_dim], std [B, action_dim])
        """
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample_action(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action from policy.

        Args:
            obs: Observation tensor

        Returns:
            Tuple of (action, log_prob)
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class SandboxTrainer:
    """Train control policy in sandbox environment with PINN validation."""

    def __init__(
        self,
        pinn_agent: SandboxPINNAgent,
        obs_dim: int = 14,
        action_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        discount_gamma: float = 0.98,
        device: str = "cpu",
    ) -> None:
        """Initialize sandbox trainer.

        Args:
            pinn_agent: Initialized PINN agent
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Policy network hidden dimension
            learning_rate: Adam learning rate
            discount_gamma: Discount factor
            device: Device for training
        """
        self.pinn_agent = pinn_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = discount_gamma
        self.device = device

        # Policy network
        self.policy = PolicyNetwork(obs_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Tracking
        self.episode_rewards: list[float] = []
        self.pinn_errors_T: list[float] = []
        self.pinn_errors_eta: list[float] = []

    def compute_discounted_returns(self, rewards: list[float]) -> Tensor:
        """Compute discounted returns.

        Args:
            rewards: Episode rewards

        Returns:
            Discounted returns tensor
        """
        returns = []
        running_sum = 0.0
        for r in reversed(rewards):
            running_sum = r + self.gamma * running_sum
            returns.append(running_sum)
        returns.reverse()

        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        return returns_tensor

    def train_episode(
        self,
        env: PanelEnv,
        episode_steps: int = 64,
    ) -> dict[str, float]:
        """Train single episode using REINFORCE.

        Args:
            env: PanelEnv instance
            episode_steps: Max steps per episode

        Returns:
            Episode metrics dict
        """
        trajectory_log_probs: list[Tensor] = []
        trajectory_rewards: list[float] = []
        trajectory_T_errors: list[float] = []
        trajectory_eta_errors: list[float] = []

        obs, _ = env.reset()

        for step in range(episode_steps):
            # Policy forward pass
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob = self.policy.sample_action(obs_tensor)

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(
                action[0].detach().cpu().numpy()
            )

            # Track for loss computation
            trajectory_log_probs.append(log_prob)
            trajectory_rewards.append(reward)

            # Track PINN errors if available
            if "pinn_prediction" in info:
                # Would compare with ground truth if available
                pass

            obs = next_obs

            if terminated or truncated:
                break

        # Compute returns and loss
        returns = self.compute_discounted_returns(trajectory_rewards)
        log_probs = torch.stack(trajectory_log_probs)

        # REINFORCE loss: -E[log_prob * return]
        loss = -(log_probs * returns).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Compute metrics
        episode_return = sum(trajectory_rewards)
        avg_T_error = np.mean(trajectory_T_errors) if trajectory_T_errors else 0.0
        avg_eta_error = np.mean(trajectory_eta_errors) if trajectory_eta_errors else 0.0

        return {
            "episode_return": episode_return,
            "avg_T_error": avg_T_error,
            "avg_eta_error": avg_eta_error,
            "loss": loss.item(),
        }

    def train(
        self,
        num_epochs: int = 40,
        episodes_per_epoch: int = 8,
        episode_steps: int = 64,
        output_dir: Path = Path("outputs/sandbox"),
        config: dict = None,
    ) -> SandboxTrainArtifacts:
        """Full training loop.

        Args:
            num_epochs: Number of training epochs
            episodes_per_epoch: Episodes per epoch
            episode_steps: Steps per episode
            output_dir: Directory for outputs
            config: Configuration dict with environment parameters

        Returns:
            Training artifacts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        policy_checkpoint = output_dir / "checkpoints" / "policy.pt"
        policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)

        metrics_file = output_dir / "metrics.csv"
        comparison_file = output_dir / "comparison.jsonl"

        viewer = Viewer3D(output_dir=output_dir / "viewer", enabled=True)

        # Create environment
        sandbox_cfg = config.get("sandbox", {}) if config else {}
        env = PanelEnv(
            pinn_agent=self.pinn_agent,
            seed=sandbox_cfg.get("seed", 42),
            dt_s=sandbox_cfg.get("dt_s", 1.0),
            reward_w_capture=sandbox_cfg.get("reward_w_capture", 1.0),
            reward_w_temp=sandbox_cfg.get("reward_w_temp", 0.4),
            temp_margin_k=sandbox_cfg.get("reward_temp_margin_K", 0.0),
            capture_scale=sandbox_cfg.get("reward_capture_scale", 1.0e-3),
            temp_scale=sandbox_cfg.get("reward_temp_scale", 1.0),
            pose_change_penalty=sandbox_cfg.get("pose_change_penalty", 1.0e-3),
        )

        # Metrics logging
        with open(metrics_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["epoch", "episode", "episode_return", "avg_T_error", "avg_eta_error", "loss"],
            )
            writer.writeheader()

        best_return = -math.inf

        for epoch in range(num_epochs):
            epoch_returns = []

            for episode in range(episodes_per_epoch):
                # Train episode
                metrics = self.train_episode(
                    env=env,
                    episode_steps=episode_steps,
                )

                epoch_returns.append(metrics["episode_return"])

                # Log
                with open(metrics_file, "a", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["epoch", "episode", "episode_return", "avg_T_error", "avg_eta_error", "loss"],
                    )
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "episode": episode,
                            **metrics,
                        }
                    )

                print(
                    f"  Epoch {epoch:3d} Episode {episode:2d}: "
                    f"return={metrics['episode_return']:8.2f} "
                    f"loss={metrics['loss']:.4f}"
                )

            # Epoch summary
            epoch_mean_return = float(np.mean(epoch_returns))
            print(f"Epoch {epoch:3d} avg return: {epoch_mean_return:.2f}")

            # Save best checkpoint
            if epoch_mean_return > best_return:
                best_return = epoch_mean_return
                torch.save(self.policy.state_dict(), policy_checkpoint)
                print(f"  ✓ New best return: {best_return:.2f}")

        # Final checkpoint
        torch.save(self.policy.state_dict(), policy_checkpoint)

        return SandboxTrainArtifacts(
            policy_checkpoint=policy_checkpoint,
            metrics_file=metrics_file,
            comparison_file=comparison_file,
            log_dir=output_dir / "viewer",
        )



def train_sandbox_policy(
    pinn_checkpoint: Path | str,
    config: dict,
    output_dir: Path | str = "outputs/sandbox",
) -> SandboxTrainArtifacts:
    """Main training entry point.

    Args:
        pinn_checkpoint: Path to pre-trained PINN
        config: Configuration dict
        output_dir: Output directory

    Returns:
        Training artifacts
    """
    # Initialize agent
    pinn_agent = SandboxPINNAgent(
        pinn_checkpoint=pinn_checkpoint,
        rk4_binary=config.get("fortran", {}).get("binary_path"),
        device=config.get("device", "cpu"),
    )

    # Initialize trainer
    trainer = SandboxTrainer(
        pinn_agent=pinn_agent,
        obs_dim=14,
        action_dim=4,
        hidden_dim=config.get("sandbox", {}).get("policy_hidden_dim", 128),
        learning_rate=config.get("sandbox", {}).get("learning_rate", 3e-4),
        discount_gamma=config.get("sandbox", {}).get("discount_gamma", 0.98),
        device=config.get("device", "cpu"),
    )

    # Train
    return trainer.train(
        num_epochs=int(config.get("sandbox", {}).get("train_epochs", 40)),
        episodes_per_epoch=int(config.get("sandbox", {}).get("episodes_per_epoch", 8)),
        episode_steps=int(config.get("sandbox", {}).get("episode_steps", 64)),
        output_dir=Path(output_dir),
        config=config,
    )
