#!/usr/bin/env python3
"""
optuna_tune.py

Usage:
  python src/PINN/sandbox/optuna_tune.py \
    --config src/PINN/configs/sandbox.yaml \
    --pinn-checkpoint src/PINN/outputs/pretrain/checkpoints/best_model.pt \
    --trials 60 \
    --episodes 12 \
    --eval-episodes 4 \
    --study-name aura_sandbox_tune \
    --storage sqlite:///optuna_aura.db
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import yaml

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]          # .../AURA-MPC
SRC_DIR = REPO_ROOT / "src"
PINN_DIR = SRC_DIR / "PINN"

for p in (str(SRC_DIR), str(PINN_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from PINN.sandbox.runtime import ClosedLoopRuntime


@dataclass
class TrialResult:
    score: float
    reward_mean: float
    delta_t_mae: float
    delta_eta_mae: float
    saturation_rate: float
    smoothness_penalty: float
    details: dict[str, Any]


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return data


def make_trial_config(base_cfg: dict[str, Any], trial: optuna.Trial) -> dict[str, Any]:
    # Focused high-leverage knobs first
    tuned = {
        "sandbox": {
            "reward_w_capture": trial.suggest_float("reward_w_capture", 0.4, 2.0, log=True),
            "reward_w_temp": trial.suggest_float("reward_w_temp", 0.05, 1.5, log=True),
            "reward_w_correction": trial.suggest_float("reward_w_correction", 0.001, 0.5, log=True),
            "correction_temp_scale_K": trial.suggest_float("correction_temp_scale_K", 5.0, 60.0),
            "correction_eta_scale": trial.suggest_float("correction_eta_scale", 0.01, 0.5, log=True),
            "correction_alpha": trial.suggest_float("correction_alpha", 0.02, 0.6),
            "pose_change_penalty": trial.suggest_float("pose_change_penalty", 1e-5, 5e-2, log=True),
            "policy_std": trial.suggest_float("policy_std", 0.05, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
            # Keep validation truthful during tuning
            "validation_mode": "every_step",
            "validation_period": 1,
        }
    }
    return deep_update(base_cfg, tuned)


def action_saturation_rate(history: list[dict[str, Any]], abs_threshold: float = 0.95) -> float:
    vals = []
    for rec in history:
        if rec.get("event") != "step":
            continue
        act = (((rec.get("policy_context") or {}).get("action_applied")) or [])
        if isinstance(act, list) and act:
            sat = any(abs(float(x)) >= abs_threshold for x in act)
            vals.append(1.0 if sat else 0.0)
    return float(sum(vals) / len(vals)) if vals else 0.0


def smoothness_proxy(history: list[dict[str, Any]]) -> float:
    # Uses pose delta if available; falls back to applied action delta.
    prev = None
    deltas = []
    for rec in history:
        if rec.get("event") != "step":
            continue
        pose = rec.get("pose") or {}
        vec = [
            float(pose.get("pitch", math.nan)),
            float(pose.get("yaw", math.nan)),
            float(pose.get("roll", math.nan)),
            float(pose.get("z", math.nan)),
        ]
        if any(math.isnan(x) for x in vec):
            ctx = rec.get("policy_context") or {}
            act = ctx.get("action_applied") or []
            if not isinstance(act, list) or len(act) < 4:
                continue
            vec = [float(act[0]), float(act[1]), float(act[2]), float(act[3])]
        if prev is not None:
            d2 = sum((a - b) ** 2 for a, b in zip(vec, prev))
            deltas.append(math.sqrt(d2))
        prev = vec
    return float(statistics.mean(deltas)) if deltas else 0.0


def extract_metrics(records: list[dict[str, Any]]) -> tuple[float, float, float]:
    rewards = []
    dts = []
    detas = []

    for rec in records:
        if rec.get("event") != "step":
            continue
        rewards.append(float(rec.get("reward", 0.0)))

        drift = rec.get("drift") or {}
        # Expected from your runtime info contract:
        # drift_dt (K), drift_deta
        if "drift_dt" in drift:
            dts.append(abs(float(drift["drift_dt"])))
        elif "dt" in drift:
            dts.append(abs(float(drift["dt"])))
        elif "delta_T" in drift:
            dts.append(abs(float(drift["delta_T"])))

        if "drift_deta" in drift:
            detas.append(abs(float(drift["drift_deta"])))
        elif "deta" in drift:
            detas.append(abs(float(drift["deta"])))
        elif "delta_eta" in drift:
            detas.append(abs(float(drift["delta_eta"])))

    reward_mean = float(statistics.mean(rewards)) if rewards else -1e9
    dt_mae = float(statistics.mean(dts)) if dts else 1e6
    deta_mae = float(statistics.mean(detas)) if detas else 1e6
    return reward_mean, dt_mae, deta_mae


def run_one_episode(runtime: ClosedLoopRuntime, learning_enabled: bool) -> list[dict[str, Any]]:
    runtime.reset()
    episode_records: list[dict[str, Any]] = []
    # Runtime/environment already knows episode length from config
    for _ in range(int(runtime.env.episode_steps)):  # type: ignore[attr-defined]
        rec = runtime.step(policy_mode="sample", learning_enabled=learning_enabled)
        episode_records.append(rec)
        if bool(rec.get("terminated", False)) or bool(rec.get("truncated", False)):
            break
    return episode_records


def run_trial_train_eval(
    cfg: dict[str, Any],
    pinn_checkpoint: Path,
    work_dir: Path,
    train_episodes: int,
    eval_episodes: int,
    seed_base: int,
    trial: optuna.Trial,
) -> TrialResult:
    out_dir = work_dir / f"trial_{trial.number:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rk4_abs = (REPO_ROOT / "src" / "RK4TRAN" / "evaluate_state").resolve()
    if not rk4_abs.exists():
        raise FileNotFoundError(f"Missing RK4 evaluator: {rk4_abs}. Build it first with src/RK4TRAN/make.sh")

    cfg = copy.deepcopy(cfg)
    cfg.setdefault("fortran", {})
    cfg["fortran"]["binary_path"] = str(rk4_abs)

    runtime = ClosedLoopRuntime(
        config=cfg,
        pinn_checkpoint=str(pinn_checkpoint),
        output_dir=str(out_dir),
        policy_checkpoint=None,
        device="cpu",
    )

    # Training phase
    for i in range(train_episodes):
        _ = run_one_episode(runtime, learning_enabled=True)
        if (i + 1) % 2 == 0:
            # Optional intermediate pruning signal
            hist = runtime.history()
            r, dt, de = extract_metrics(hist[-max(1, len(hist)//2):])
            interim = r - (0.02 * dt) - (8.0 * de)
            trial.report(interim, step=i + 1)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Eval phase (no learning)
    eval_rewards = []
    eval_dts = []
    eval_detas = []
    sat_rates = []
    smooth_vals = []

    for j in range(eval_episodes):
        # Reseed env reproducibly across trials
        runtime.reset(seed=seed_base + j)
        records = run_one_episode(runtime, learning_enabled=False)
        r, dt, de = extract_metrics(records)
        eval_rewards.append(r)
        eval_dts.append(dt)
        eval_detas.append(de)
        sat_rates.append(action_saturation_rate(records))
        smooth_vals.append(smoothness_proxy(records))

    reward_mean = float(statistics.mean(eval_rewards))
    delta_t_mae = float(statistics.mean(eval_dts))
    delta_eta_mae = float(statistics.mean(eval_detas))
    saturation_rate = float(statistics.mean(sat_rates))
    smoothness_pen = float(statistics.mean(smooth_vals))

    # Scalar objective (maximize)
    # Tune weights for your preference:
    w_dt = 0.03
    w_deta = 10.0
    w_sat = 1.0
    w_smooth = 0.2
    score = reward_mean - (w_dt * delta_t_mae) - (w_deta * delta_eta_mae) - (w_sat * saturation_rate) - (w_smooth * smoothness_pen)

    details = {
        "reward_mean": reward_mean,
        "delta_t_mae": delta_t_mae,
        "delta_eta_mae": delta_eta_mae,
        "saturation_rate": saturation_rate,
        "smoothness_penalty": smoothness_pen,
        "trace_path": str(out_dir / "closed_loop_trace.jsonl"),
    }

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(details, f, indent=2)

    return TrialResult(
        score=score,
        reward_mean=reward_mean,
        delta_t_mae=delta_t_mae,
        delta_eta_mae=delta_eta_mae,
        saturation_rate=saturation_rate,
        smoothness_penalty=smoothness_pen,
        details=details,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pinn-checkpoint", type=Path, required=True)
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--episodes", type=int, default=12, help="train episodes per trial")
    p.add_argument("--eval-episodes", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--study-name", type=str, default="aura_sandbox_tune")
    p.add_argument("--storage", type=str, default="sqlite:///optuna_aura.db")
    p.add_argument("--out-dir", type=Path, default=Path("src/PINN/outputs/optuna"))
    p.add_argument("--n-jobs", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(args.config)

    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: optuna.Trial) -> float:
        cfg = make_trial_config(base_cfg, trial)

        result = run_trial_train_eval(
            cfg=cfg,
            pinn_checkpoint=args.pinn_checkpoint,
            work_dir=args.out_dir,
            train_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            seed_base=args.seed_base,
            trial=trial,
        )

        trial.set_user_attr("reward_mean", result.reward_mean)
        trial.set_user_attr("delta_t_mae", result.delta_t_mae)
        trial.set_user_attr("delta_eta_mae", result.delta_eta_mae)
        trial.set_user_attr("saturation_rate", result.saturation_rate)
        trial.set_user_attr("smoothness_penalty", result.smoothness_penalty)
        trial.set_user_attr("details", result.details)

        return result.score

    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs, gc_after_trial=True)

    best = study.best_trial
    print("\n=== Best trial ===")
    print(f"number: {best.number}")
    print(f"value:  {best.value:.6f}")
    print("params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("metrics:")
    for k in ["reward_mean", "delta_t_mae", "delta_eta_mae", "saturation_rate", "smoothness_penalty"]:
        if k in best.user_attrs:
            print(f"  {k}: {best.user_attrs[k]}")

    # Write best config snapshot
    best_cfg = make_trial_config(base_cfg, best)
    best_cfg_path = args.out_dir / "best_sandbox.yaml"
    with best_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)
    print(f"\nBest config written to: {best_cfg_path}")


if __name__ == "__main__":
    main()