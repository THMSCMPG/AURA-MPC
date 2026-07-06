"""CLI for sandbox training, validation, live bridge, and Python 3D viewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ...config import load_config
from .live import JsonlSensorStream, run_live_stream
from .training import train_policy, validate_against_fortran
from .viewer import run_viewer


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PINN sandbox orchestrator")
    p.add_argument("--config", type=Path, default=Path("configs/sandbox.yaml"))
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("train", help="Train RL policy in 3D sandbox")
    sp.add_argument("--out-dir", type=Path, default=Path("results/sandbox"))
    sp.add_argument("--validate", action="store_true")
    sp.add_argument("--aura-root", type=Path, default=None)

    sp = sub.add_parser("validate", help="Validate sandbox metrics against Fortran tiers")
    sp.add_argument("--aura-root", type=Path, default=None)
    sp.add_argument(
        "--report",
        type=Path,
        default=Path("results/sandbox/fortran_validation.json"),
    )

    sp = sub.add_parser("live", help="Run live mock stream -> orchestrator command JSONL")
    sp.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Phase-2 sandbox policy checkpoint used to seed control state.",
    )
    sp.add_argument(
        "--orchestrator-checkpoint",
        type=Path,
        default=None,
        help="DualHeadPINN checkpoint for RealTimeOrchestrator model inference.",
    )
    sp.add_argument("--aura-root", type=Path, default=None)
    sp.add_argument("--stream-jsonl", type=Path, required=True)
    sp.add_argument("--output-jsonl", type=Path, default=None)
    sp.add_argument("--realtime", action="store_true")

    sp = sub.add_parser("viewer", help="Render Python 3D twin")
    sp.add_argument("--mode", choices=("auto", "manual", "live"), default="live")
    sp.add_argument("--commands", type=Path, default=Path("results/commands.jsonl"))
    sp.add_argument("--output", type=Path, default=None)
    sp.add_argument("--frames", type=int, default=120)
    sp.add_argument("--fps", type=int, default=24)
    sp.add_argument("--height-mm", type=float, default=65.0)
    sp.add_argument("--yaw", type=float, default=0.0)
    sp.add_argument("--pitch", type=float, default=0.0)
    sp.add_argument("--roll", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = load_config(args.config if args.config.exists() else None)

    if args.command == "train":
        artifacts = train_policy(cfg, args.out_dir)
        print(
            json.dumps(
                {
                    "checkpoint": str(artifacts.checkpoint_path),
                    "metrics": str(artifacts.metrics_path),
                    "plot": str(artifacts.plot_path),
                },
                indent=2,
            )
        )
        if args.validate:
            report_path = args.out_dir / "fortran_validation.json"
            report = validate_against_fortran(cfg, args.aura_root, report_path)
            print(json.dumps(report, indent=2))
        return 0

    if args.command == "validate":
        report = validate_against_fortran(cfg, args.aura_root, args.report)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "live":
        out_jsonl = (
            Path(args.output_jsonl)
            if args.output_jsonl is not None
            else Path(cfg.sandbox.command_log_path)
        )
        stream = JsonlSensorStream(Path(args.stream_jsonl))
        n = run_live_stream(
            cfg,
            orchestrator_checkpoint=(
                Path(args.orchestrator_checkpoint)
                if args.orchestrator_checkpoint is not None
                else None
            ),
            policy_checkpoint=Path(args.checkpoint) if args.checkpoint is not None else None,
            stream=stream,
            output_jsonl=out_jsonl,
            aura_mfp_root=Path(args.aura_root) if args.aura_root is not None else None,
            realtime=bool(args.realtime),
        )
        print(json.dumps({"commands_written": n, "output_jsonl": str(out_jsonl)}, indent=2))
        return 0

    run_viewer(
        mode=args.mode,
        command_path=Path(args.commands),
        output_path=args.output,
        frames=max(1, int(args.frames)),
        fps=max(1, int(args.fps)),
        manual_pose={
            "height": float(args.height_mm),
            "yaw": float(args.yaw),
            "pitch": float(args.pitch),
            "roll": float(args.roll),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
