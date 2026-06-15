"""Policy training + Fortran-tier validation for the sandbox environment."""

from __future__ import annotations

import json
import math
import subprocess  # noqa: S404 - required for local tier-binary checks
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import matplotlib
import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ...config import AppConfig
from ..orchestrator import _binary_path
from ..physics import sandia_efficiency
from .environment import PanelEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True, slots=True)
class TrainArtifacts:
    """Output bundle for sandbox training."""

    checkpoint_path: Path
    metrics_path: Path
    plot_path: Path


class PolicyNet(nn.Module):
    """Compact Gaussian policy for continuous 4-DoF control."""

    def __init__(self, obs_dim: int, hidden: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return cast(Tensor, self.net(obs))


def _discounted_returns(rewards: list[float], gamma: float) -> Tensor:
    out: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    t = torch.tensor(out, dtype=torch.float32)
    if t.numel() == 0:
        return t
    return (t - t.mean()) / (t.std(unbiased=False) + 1e-8)


def train_policy(cfg: AppConfig, out_dir: Path) -> TrainArtifacts:
    """Train a sandbox policy with REINFORCE and write checkpoint + metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = PanelEnv(cfg)
    policy = PolicyNet(env.observation_dim, cfg.sandbox.policy_hidden_dim, env.action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.sandbox.learning_rate)
    std = torch.tensor(float(cfg.sandbox.policy_std), dtype=torch.float32)

    reward_curve: list[float] = []
    best_reward = -math.inf
    best_state: dict[str, Tensor] | None = None

    for epoch in range(int(cfg.sandbox.train_epochs)):
        epoch_rewards: list[float] = []
        epoch_loss = torch.tensor(0.0)
        for ep in range(int(cfg.sandbox.episodes_per_epoch)):
            obs_np, _ = env.reset(seed=cfg.sandbox.seed + epoch * 1000 + ep)
            done = False
            log_probs: list[Tensor] = []
            rewards: list[float] = []
            while not done:
                obs = torch.from_numpy(obs_np).float().unsqueeze(0)
                mean = policy(obs).squeeze(0)
                dist = Normal(mean, std.expand_as(mean))
                raw_action = dist.rsample()
                action = torch.tanh(raw_action)
                log_prob = dist.log_prob(raw_action).sum()  # type: ignore[no-untyped-call]
                obs_np, reward, terminated, truncated, _ = env.step(
                    action.detach().cpu().numpy()
                )
                log_probs.append(log_prob)
                rewards.append(float(reward))
                done = bool(terminated or truncated)
            returns = _discounted_returns(rewards, float(cfg.sandbox.discount_gamma))
            if returns.numel() > 0:
                ep_loss = torch.tensor(0.0)
                for lp, ret in zip(log_probs, returns):
                    ep_loss = ep_loss - lp * ret
                epoch_loss = epoch_loss + ep_loss / returns.numel()
            epoch_rewards.append(float(sum(rewards)))

        optimizer.zero_grad(set_to_none=True)
        epoch_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        mean_reward = float(sum(epoch_rewards) / max(1, len(epoch_rewards)))
        reward_curve.append(mean_reward)
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

    if best_state is not None:
        policy.load_state_dict(best_state)

    checkpoint_path = out_dir / "sandbox_policy.pt"
    metrics_path = out_dir / "sandbox_training_metrics.json"
    plot_path = out_dir / "sandbox_reward_curve.png"

    torch.save(
        {
            "policy_state": policy.state_dict(),
            "obs_dim": env.observation_dim,
            "action_dim": env.action_dim,
            "config": {
                "seed": cfg.sandbox.seed,
                "beta_Pmax": cfg.physics.beta_Pmax,
                "eta_ref": cfg.physics.eta_ref,
                "T_ref_K": cfg.physics.T_ref_K,
            },
            "reward_curve": reward_curve,
            "best_reward": best_reward,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
        checkpoint_path,
    )

    metrics_path.write_text(
        json.dumps(
            {
                "reward_curve": reward_curve,
                "best_reward": best_reward,
                "epochs": cfg.sandbox.train_epochs,
                "episodes_per_epoch": cfg.sandbox.episodes_per_epoch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(reward_curve, color="tab:blue")
    ax.set_title("Sandbox RL reward curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return TrainArtifacts(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        plot_path=plot_path,
    )


def _extract_fortran_temperature(result: dict[str, Any]) -> float | None:
    for key in ("T_panel", "T_mod", "temperature", "predicted_temp"):
        if key in result:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                return None
    return None


def validate_against_fortran(
    cfg: AppConfig,
    aura_root: Path | None,
    out_path: Path,
) -> dict[str, Any]:
    """Compare sandbox temperatures/efficiency against available tier binaries."""
    env = PanelEnv(cfg)
    tiers = ("SIMV1", "SIMV2", "SIMV3", "SIMV4")
    samples = int(cfg.sandbox.validation_samples)
    report: dict[str, Any] = {"samples": samples, "tiers": {}}
    if aura_root is None:
        report["warning"] = "aura_mfp_root not provided; validation skipped"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    for tier in tiers:
        binary = _binary_path(aura_root, tier)
        if not binary.exists():
            report["tiers"][tier] = {"status": "missing_binary", "path": str(binary)}
            continue

        temp_errs: list[float] = []
        eta_errs: list[float] = []
        failures = 0
        for i in range(samples):
            env.reset(seed=cfg.sandbox.seed + i)
            action = torch.zeros(4, dtype=torch.float32).numpy()
            _, _, _, _, info = env.step(action)
            pose = info["pose"]
            metrics = info["metrics"]
            packet = {
                "t_s": float(env.conditions.hour * 3600.0),
                "G_poa": float(metrics["G_poa"]),
                "T_amb": float(env.conditions.ambient_c),
                "WS": float(env.conditions.wind_mps),
                "CC": float(env.conditions.cloud_cover),
                "lat": float(env.conditions.lat),
                "lon": float(env.conditions.lon),
                "pose": {k: float(pose[k]) for k in ("pitch", "yaw", "roll", "z")},
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "img_features": [],
            }
            proc = subprocess.run(  # noqa: S603
                [str(binary)],
                input=json.dumps(packet),
                capture_output=True,
                text=True,
                check=False,
                timeout=12,
            )
            if proc.returncode != 0:
                failures += 1
                continue
            try:
                parsed = json.loads(proc.stdout)
            except json.JSONDecodeError:
                failures += 1
                continue
            if not isinstance(parsed, dict):
                failures += 1
                continue
            fortran_temp_c = _extract_fortran_temperature(parsed)
            if fortran_temp_c is None:
                failures += 1
                continue

            sandbox_temp_c = metrics["T_panel_K"] - 273.15
            temp_errs.append(abs(sandbox_temp_c - fortran_temp_c))
            eta_ref = torch.tensor(cfg.physics.eta_ref, dtype=torch.float32)
            beta = torch.tensor(cfg.physics.beta_Pmax, dtype=torch.float32)
            tref = torch.tensor(cfg.physics.T_ref_K, dtype=torch.float32)
            eta_fortran = float(
                sandia_efficiency(
                    T_K=torch.tensor(fortran_temp_c + 273.15, dtype=torch.float32),
                    eta_ref=eta_ref,
                    beta_Pmax=beta,
                    T_ref_K=tref,
                ).item()
            )
            eta_errs.append(abs(float(metrics["eta"]) - eta_fortran))

        n = len(temp_errs)
        if n == 0:
            report["tiers"][tier] = {
                "status": "no_comparable_outputs",
                "failures": failures,
            }
            continue
        report["tiers"][tier] = {
            "status": "ok",
            "n": n,
            "failures": failures,
            "mae_temp_c": float(sum(temp_errs) / n),
            "rmse_temp_c": float(math.sqrt(sum(e * e for e in temp_errs) / n)),
            "mae_eta": float(sum(eta_errs) / n),
            "rmse_eta": float(math.sqrt(sum(e * e for e in eta_errs) / n)),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
