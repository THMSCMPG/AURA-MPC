"""Sandbox RL training for the RK4TRAN-validated closed loop."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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
    trajectories_dir: Path


class PolicyNetwork(nn.Module):
    """Gaussian policy for continuous 4-DoF control."""

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, std_init: float = 0.25) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(std_init))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def sample_action(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        squashed = torch.tanh(action)
        return squashed, log_prob, mean

    def mean_action(self, obs: Tensor) -> Tensor:
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SandboxTrainer:
    """Train the control policy using RK4TRAN-validated rewards."""

    def __init__(
        self,
        pinn_agent: SandboxPINNAgent,
        obs_dim: int = 18,
        action_dim: int = 4,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        discount_gamma: float = 0.98,
        policy_std: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.pinn_agent = pinn_agent
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = float(discount_gamma)
        self.device = device
        self.policy = PolicyNetwork(obs_dim, hidden_dim, action_dim, std_init=policy_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)

    def compute_discounted_returns(self, rewards: list[float]) -> Tensor:
        returns: list[float] = []
        running_sum = 0.0
        for reward in reversed(rewards):
            running_sum = reward + self.gamma * running_sum
            returns.append(running_sum)
        returns.reverse()
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns_tensor.numel() > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        return returns_tensor

    def _serialize_step_record(
        self,
        *,
        epoch: int,
        episode: int,
        step: int,
        reward: float,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "epoch": int(epoch),
            "episode": int(episode),
            "step": int(step),
            "reward": float(reward),
            **info,
        }

    def train_episode(
        self,
        env: PanelEnv,
        episode_steps: int,
        epoch_idx: int,
        episode_idx: int,
        viewer: Viewer3D | None = None,
    ) -> dict[str, Any]:
        trajectory_log_probs: list[Tensor] = []
        trajectory_rewards: list[float] = []
        step_records: list[dict[str, Any]] = []

        obs, reset_info = env.reset(seed=env.seed + epoch_idx * 1000 + episode_idx)
        if viewer is not None:
            viewer.render_state(
                pose=reset_info["pose"],
                weather=reset_info["inputs"]["weather"],
                pinn_pred=reset_info["pinn_prediction"],
                rk4_pred=reset_info["rk4_prediction"],
                reward_breakdown=reset_info["reward_breakdown"],
                discrepancy=reset_info["discrepancy"],
                decision_reason=reset_info["decision_reason"],
                episode=episode_idx,
                step=-1,
            )

        for step_idx in range(episode_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, mean = self.policy.sample_action(obs_tensor)
            action_np = action[0].detach().cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(
                action_np,
                policy_context={
                    "mode": "sample",
                    "action_mean": mean[0].detach().cpu().tolist(),
                    "action_applied": action_np.tolist(),
                    "log_prob": float(log_prob.item()),
                },
            )
            trajectory_log_probs.append(log_prob.squeeze())
            trajectory_rewards.append(float(reward))
            step_record = self._serialize_step_record(
                epoch=epoch_idx,
                episode=episode_idx,
                step=step_idx,
                reward=reward,
                info=info,
            )
            step_records.append(step_record)

            if viewer is not None:
                viewer.render_state(
                    pose=info["pose"],
                    weather=info["inputs"]["weather"],
                    pinn_pred=info["pinn_prediction"],
                    rk4_pred=info["rk4_prediction"],
                    reward_breakdown=info["reward_breakdown"],
                    discrepancy=info["discrepancy"],
                    decision_reason=info["decision_reason"],
                    episode=episode_idx,
                    step=step_idx,
                )

            obs = next_obs
            if terminated or truncated:
                break

        returns = self.compute_discounted_returns(trajectory_rewards)
        log_probs = torch.stack(trajectory_log_probs)
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        t_errors = [
            abs(float(record["discrepancy"]["T_operating"]))
            for record in step_records
            if record.get("rk4_prediction") is not None
        ]
        eta_errors = [
            abs(float(record["discrepancy"]["eta"]))
            for record in step_records
            if record.get("rk4_prediction") is not None
        ]

        if viewer is not None:
            viewer.plot_episode_summary(
                episode=episode_idx,
                rewards=trajectory_rewards,
                T_errors=t_errors,
                eta_errors=eta_errors,
            )

        return {
            "episode_return": float(sum(trajectory_rewards)),
            "avg_T_error": float(np.mean(t_errors)) if t_errors else 0.0,
            "avg_eta_error": float(np.mean(eta_errors)) if eta_errors else 0.0,
            "loss": float(loss.item()),
            "step_records": step_records,
        }

    def _load_policy_state(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
            state = checkpoint["policy_state"]
        else:
            state = checkpoint
        if isinstance(state, dict):
            self.policy.load_state_dict(state)

    def train(
        self,
        num_epochs: int = 40,
        episodes_per_epoch: int = 8,
        episode_steps: int = 64,
        output_dir: Path = Path("outputs/sandbox"),
        config: dict[str, Any] | None = None,
    ) -> SandboxTrainArtifacts:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        policy_checkpoint = output_dir / "checkpoints" / "policy.pt"
        policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = output_dir / "metrics.csv"
        comparison_file = output_dir / "comparison.jsonl"
        trajectories_dir = output_dir / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        viewer = Viewer3D(output_dir=output_dir / "viewer", enabled=True)

        sandbox_cfg = config.get("sandbox", {}) if config else {}
        condition_bounds = {
            "lat_min": sandbox_cfg.get("lat_min", -65.0),
            "lat_max": sandbox_cfg.get("lat_max", 65.0),
            "lon_min": sandbox_cfg.get("lon_min", -180.0),
            "lon_max": sandbox_cfg.get("lon_max", 180.0),
            "day_min": sandbox_cfg.get("day_min", 1),
            "day_max": sandbox_cfg.get("day_max", 365),
            "hour_min": sandbox_cfg.get("hour_min", 5.0),
            "hour_max": sandbox_cfg.get("hour_max", 19.0),
            "ambient_c_min": sandbox_cfg.get("ambient_c_min", -15.0),
            "ambient_c_max": sandbox_cfg.get("ambient_c_max", 45.0),
            "wind_min": sandbox_cfg.get("wind_min", 0.0),
            "wind_max": sandbox_cfg.get("wind_max", 20.0),
            "cloud_min": sandbox_cfg.get("cloud_min", 0.0),
            "cloud_max": sandbox_cfg.get("cloud_max", 1.0),
            "g_peak_min": sandbox_cfg.get("g_peak_min", 500.0),
            "g_peak_max": sandbox_cfg.get("g_peak_max", 1100.0),
        }
        env = PanelEnv(
            pinn_agent=self.pinn_agent,
            seed=int(sandbox_cfg.get("seed", 42)),
            dt_s=float(sandbox_cfg.get("dt_s", 1.0)),
            episode_steps=int(sandbox_cfg.get("episode_steps", 64)),
            reward_w_capture=float(sandbox_cfg.get("reward_w_capture", 1.0)),
            reward_w_temp=float(sandbox_cfg.get("reward_w_temp", 0.4)),
            reward_w_correction=float(sandbox_cfg.get("reward_w_correction", 0.05)),
            temp_margin_k=float(sandbox_cfg.get("reward_temp_margin_K", 0.0)),
            capture_scale=float(sandbox_cfg.get("reward_capture_scale", 1.0e-3)),
            temp_scale=float(sandbox_cfg.get("reward_temp_scale", 1.0)),
            pose_change_penalty=float(sandbox_cfg.get("pose_change_penalty", 1.0e-3)),
            correction_temp_scale_k=float(sandbox_cfg.get("correction_temp_scale_K", 15.0)),
            correction_eta_scale=float(sandbox_cfg.get("correction_eta_scale", 0.05)),
            validation_mode=str(sandbox_cfg.get("validation_mode", "every_step")),
            validation_period=int(sandbox_cfg.get("validation_period", 1)),
            reward_source=str(sandbox_cfg.get("reward_source", "validated")),
            condition_bounds=condition_bounds,
        )

        with metrics_file.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["epoch", "episode", "episode_return", "avg_T_error", "avg_eta_error", "loss"],
            )
            writer.writeheader()
        comparison_file.write_text("", encoding="utf-8")

        best_return = -math.inf

        for epoch in range(num_epochs):
            epoch_returns: list[float] = []
            for episode in range(episodes_per_epoch):
                metrics = self.train_episode(
                    env=env,
                    episode_steps=episode_steps,
                    epoch_idx=epoch,
                    episode_idx=episode,
                    viewer=viewer,
                )
                epoch_returns.append(metrics["episode_return"])

                with metrics_file.open("a", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(
                        fh,
                        fieldnames=["epoch", "episode", "episode_return", "avg_T_error", "avg_eta_error", "loss"],
                    )
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "episode": episode,
                            "episode_return": metrics["episode_return"],
                            "avg_T_error": metrics["avg_T_error"],
                            "avg_eta_error": metrics["avg_eta_error"],
                            "loss": metrics["loss"],
                        }
                    )

                episode_trace = trajectories_dir / f"epoch_{epoch:03d}_episode_{episode:03d}.jsonl"
                with episode_trace.open("w", encoding="utf-8") as trace_fh:
                    for record in metrics["step_records"]:
                        line = json.dumps(record, sort_keys=True)
                        trace_fh.write(line + "\n")
                        with comparison_file.open("a", encoding="utf-8") as comparison_fh:
                            comparison_fh.write(line + "\n")

                print(
                    f"  Epoch {epoch:3d} Episode {episode:2d}: "
                    f"return={metrics['episode_return']:8.2f} "
                    f"T_err={metrics['avg_T_error']:.3f} "
                    f"eta_err={metrics['avg_eta_error']:.5f} "
                    f"loss={metrics['loss']:.4f}"
                )

            epoch_mean_return = float(np.mean(epoch_returns))
            print(f"Epoch {epoch:3d} avg return: {epoch_mean_return:.2f}")

            if epoch_mean_return > best_return:
                best_return = epoch_mean_return
                torch.save(
                    {
                        "policy_state": self.policy.state_dict(),
                        "obs_dim": self.obs_dim,
                        "action_dim": self.action_dim,
                        "best_return": best_return,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    },
                    policy_checkpoint,
                )
                print(f"  ✓ New best return: {best_return:.2f}")

        torch.save(
            {
                "policy_state": self.policy.state_dict(),
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "best_return": best_return,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            policy_checkpoint,
        )

        return SandboxTrainArtifacts(
            policy_checkpoint=policy_checkpoint,
            metrics_file=metrics_file,
            comparison_file=comparison_file,
            log_dir=output_dir / "viewer",
            trajectories_dir=trajectories_dir,
        )


def train_sandbox_policy(
    pinn_checkpoint: Path | str,
    config: dict[str, Any],
    output_dir: Path | str = "outputs/sandbox",
) -> SandboxTrainArtifacts:
    pinn_checkpoint = Path(pinn_checkpoint)
    normalizer_path = pinn_checkpoint.parent / "normalizer.json"
    pinn_agent = SandboxPINNAgent(
        pinn_checkpoint=pinn_checkpoint,
        rk4_binary=config.get("fortran", {}).get("binary_path"),
        normalizer_path=normalizer_path if normalizer_path.exists() else None,
        device=config.get("device", "cpu"),
        correction_alpha=float(config.get("sandbox", {}).get("correction_alpha", 0.25)),
    )

    trainer = SandboxTrainer(
        pinn_agent=pinn_agent,
        obs_dim=18,
        action_dim=4,
        hidden_dim=int(config.get("sandbox", {}).get("policy_hidden_dim", 128)),
        learning_rate=float(config.get("sandbox", {}).get("learning_rate", 3e-4)),
        discount_gamma=float(config.get("sandbox", {}).get("discount_gamma", 0.98)),
        policy_std=float(config.get("sandbox", {}).get("policy_std", 0.25)),
        device=config.get("device", "cpu"),
    )

    return trainer.train(
        num_epochs=int(config.get("sandbox", {}).get("train_epochs", 40)),
        episodes_per_epoch=int(config.get("sandbox", {}).get("episodes_per_epoch", 8)),
        episode_steps=int(config.get("sandbox", {}).get("episode_steps", 64)),
        output_dir=Path(output_dir),
        config=config,
    )
