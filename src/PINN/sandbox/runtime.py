"""Interactive closed-loop runtime for MATLAB and replay-style demos."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .environment import PanelEnv
from .integration import SandboxPINNAgent
from .training import PolicyNetwork


class ClosedLoopRuntime:
    """Stateful runtime for stepping the control loop one decision at a time."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        pinn_checkpoint: Path | str,
        output_dir: Path | str,
        policy_checkpoint: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.output_dir / "closed_loop_trace.jsonl"
        self.trace_path.write_text("", encoding="utf-8")

        pinn_checkpoint = Path(pinn_checkpoint)
        normalizer_path = pinn_checkpoint.parent / "normalizer.json"
        sandbox_cfg = config.get("sandbox", {})
        self.pinn_agent = SandboxPINNAgent(
            pinn_checkpoint=pinn_checkpoint,
            rk4_binary=config.get("fortran", {}).get("binary_path"),
            normalizer_path=normalizer_path if normalizer_path.exists() else None,
            device=device,
            correction_alpha=float(sandbox_cfg.get("correction_alpha", 0.25)),
        )

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
        self.env = PanelEnv(
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
        self.policy = PolicyNetwork(
            self.env.observation_dim,
            int(sandbox_cfg.get("policy_hidden_dim", 128)),
            self.env.action_dim,
            std_init=float(sandbox_cfg.get("policy_std", 0.25)),
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=float(sandbox_cfg.get("learning_rate", 3e-4)),
        )
        self.discount_gamma = float(sandbox_cfg.get("discount_gamma", 0.98))
        self.auto_update_each_episode = bool(sandbox_cfg.get("online_policy_updates", True))
        self._episode_log_probs: list[torch.Tensor] = []
        self._episode_rewards: list[float] = []
        self._history: list[dict[str, Any]] = []
        self._current_obs, self._latest_info = self.env.reset()

        if policy_checkpoint is not None:
            self.load_policy(policy_checkpoint)

    def load_policy(self, policy_checkpoint: Path | str) -> None:
        checkpoint = torch.load(Path(policy_checkpoint), map_location=self.device)
        if isinstance(checkpoint, dict) and "policy_state" in checkpoint:
            state = checkpoint["policy_state"]
        else:
            state = checkpoint
        if not isinstance(state, dict):
            raise TypeError("Policy checkpoint must contain a state dict")
        self.policy.load_state_dict(state)

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.discount_gamma * running
            returns.append(running)
        returns.reverse()
        tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if tensor.numel() > 1:
            tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
        return tensor

    def _append_trace(self, record: dict[str, Any]) -> None:
        self._history.append(record)
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def reset(
        self,
        *,
        initial_conditions: dict[str, Any] | None = None,
        initial_pose: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        self._episode_log_probs.clear()
        self._episode_rewards.clear()
        self._history.clear()
        self.trace_path.write_text("", encoding="utf-8")
        self._current_obs, self._latest_info = self.env.reset(
            seed=seed,
            conditions=initial_conditions,
            pose=initial_pose,
        )
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "reset",
            **self._latest_info,
            "policy_updated": False,
        }
        self._append_trace(record)
        return record

    def update_policy(self) -> bool:
        if not self._episode_log_probs or not self._episode_rewards:
            return False
        log_probs = torch.stack(self._episode_log_probs)
        returns = self._discounted_returns(self._episode_rewards)
        loss = -(log_probs * returns).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        self._episode_log_probs.clear()
        self._episode_rewards.clear()
        return True

    def step(
        self,
        *,
        action_override: list[float] | None = None,
        policy_mode: str = "mean",
        learning_enabled: bool = True,
        validate_with_rk4: bool | None = None,
    ) -> dict[str, Any]:
        obs_tensor = torch.tensor(self._current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if action_override is not None:
            action_tensor = torch.tensor(action_override, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = torch.clamp(action_tensor, -1.0, 1.0)
            log_prob = None
            mean = action
            mode = "manual"
        elif policy_mode == "sample":
            action, log_prob, mean = self.policy.sample_action(obs_tensor)
            mode = "sample"
        else:
            mean = self.policy.mean_action(obs_tensor)
            action = mean
            log_prob = None
            mode = "mean"

        action_np = action[0].detach().cpu().numpy()
        next_obs, reward, terminated, truncated, info = self.env.step(
            action_np,
            policy_context={
                "mode": mode,
                "action_mean": mean[0].detach().cpu().tolist(),
                "action_applied": action_np.tolist(),
                "learning_enabled": bool(learning_enabled),
            },
            validate_with_rk4=validate_with_rk4,
        )
        self._current_obs = next_obs
        self._latest_info = info

        if learning_enabled and log_prob is not None:
            self._episode_log_probs.append(log_prob.squeeze())
            self._episode_rewards.append(float(reward))

        policy_updated = False
        if learning_enabled and self.auto_update_each_episode and (terminated or truncated):
            policy_updated = self.update_policy()

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "step",
            "reward": float(reward),
            **info,
            "policy_updated": policy_updated,
        }
        self._append_trace(record)
        return record

    def inject_conditions(self, conditions: dict[str, Any]) -> None:
        """Overwrite the environment's conditions ahead of the next :meth:`step`.

        This is the entry point EDGE-driven operation (or a MATLAB "Use
        EDGE feed" toggle) uses to hand a live sensor reading — already
        translated by :func:`sandbox.edge_adapter.edge_packet_to_conditions`
        — into a *running* episode. It does not reset pose or history;
        it only changes what the next :meth:`step` will observe.
        """
        self.env.set_conditions(conditions)

    def latest(self) -> dict[str, Any]:
        return dict(self._latest_info)

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
