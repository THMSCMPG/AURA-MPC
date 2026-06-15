"""Python 3D viewer for the AURA 4-DOF digital twin (MATLAB-inspired)."""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import numpy.typing as npt

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # type: ignore[import-untyped]  # noqa: E402

FloatArray = npt.NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class TwinGeometry:
    BASE_THICK: float = 10.0
    LOWER_H: float = 220.0
    UPPER_H: float = 160.0
    UPPER_TRAVEL: float = 130.0
    YAW_THICK: float = 16.0
    ARM_H: float = 55.0
    ARM_BASE_THICK: float = 8.0
    PIVOT_BOSS_D: float = 12.0
    PM_THICK: float = 6.0
    PM_LUG_H: float = 14.0
    PANEL_Z: float = 70.0
    YAW_SEAT_Z: float = 88.0
    H_MAX: float = 130.0
    YAW_LIM: float = 180.0
    PITCH_LIM: float = 35.0
    ROLL_LIM: float = 25.0


def map_command_to_twin(command: dict[str, Any], z_max_m: float = 3.0) -> dict[str, float]:
    """Map orchestrator fields to twin kinematics.

    Uses linear mapping: ``height_mm = clamp(z_m,0,z_max) / z_max * 130``.
    """
    if "z" in command:
        z_m = float(command.get("z", 0.0))
        z_clamped = min(max(z_m, 0.0), z_max_m)
        h_mm = 130.0 * (z_clamped / max(1e-6, z_max_m))
    else:
        h_mm = float(command.get("height", command.get("height_mm", 0.0)))
    return {
        "height": h_mm,
        "yaw": float(command.get("yaw", 0.0)),
        "pitch": float(command.get("pitch", 0.0)),
        "roll": float(command.get("roll", 0.0)),
    }


def read_latest_command(path: Path) -> dict[str, Any] | None:
    """Read the last JSON object from a JSONL file using an EOF tail scan."""
    if not path.exists():
        return None
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            if size == 0:
                return None
            block = 4096
            start = max(0, size - block)
            fh.seek(start, os.SEEK_SET)
            buf = fh.read(size - start)
        lines = [ln for ln in buf.splitlines() if ln.strip()]
        if not lines:
            return None
        raw = lines[-1].decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_all_commands(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    records.append(parsed)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _box(cx: float, cy: float, cz: float, sx: float, sy: float, sz: float) -> list[FloatArray]:
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    z0, z1 = cz - sz / 2.0, cz + sz / 2.0
    v = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    idx = (
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (1, 2, 6, 5),
        (0, 3, 7, 4),
    )
    return [v[list(i)] for i in idx]


def _rot_z(deg: float) -> FloatArray:
    r = math.radians(deg)
    return np.array(
        [[math.cos(r), -math.sin(r), 0.0], [math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rot_x(deg: float) -> FloatArray:
    r = math.radians(deg)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(r), -math.sin(r)], [0.0, math.sin(r), math.cos(r)]],
        dtype=np.float64,
    )


def _rot_y(deg: float) -> FloatArray:
    r = math.radians(deg)
    return np.array(
        [[math.cos(r), 0.0, math.sin(r)], [0.0, 1.0, 0.0], [-math.sin(r), 0.0, math.cos(r)]],
        dtype=np.float64,
    )


def _xf(verts: FloatArray, rot: FloatArray, t: FloatArray) -> FloatArray:
    transformed = (rot @ verts.T).T + t
    return np.asarray(transformed, dtype=np.float64)


def render_frame(ax: Any, pose: dict[str, float], geom: TwinGeometry, mode: str) -> None:
    ax.clear()
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(0, 700)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=20, azim=135)

    h = pose["height"]
    yaw = pose["yaw"]
    pitch = pose["pitch"]
    roll = pose["roll"]

    base = _box(0, 0, geom.BASE_THICK / 2.0, 220, 220, geom.BASE_THICK)
    post = _box(0, 0, geom.BASE_THICK + geom.LOWER_H / 2.0, 30, 30, geom.LOWER_H)
    inner_center_z = geom.BASE_THICK + geom.LOWER_H + 43.8 + h
    inner = _box(0, 0, inner_center_z, 20, 20, geom.UPPER_H)

    yaw_center = np.array([0.0, 0.0, inner_center_z + geom.YAW_SEAT_Z], dtype=np.float64)
    yaw_disk = _box(yaw_center[0], yaw_center[1], yaw_center[2], 70, 70, geom.YAW_THICK)
    ryaw = _rot_z(yaw)
    yaw_disk = [_xf(face, ryaw, np.zeros(3)) for face in yaw_disk]

    arm_origin = yaw_center + np.array([0.0, 0.0, geom.ARM_BASE_THICK + geom.ARM_H], dtype=np.float64)
    rp = ryaw @ _rot_x(pitch)
    rr = rp @ _rot_y(roll)
    panel_center = arm_origin + rr @ np.array([0.0, 0.0, geom.PANEL_Z], dtype=np.float64)
    panel = _box(0, 0, 0, 260, 180, geom.PM_THICK)
    panel = [_xf(face, rr, panel_center) for face in panel]

    def add(faces: list[FloatArray], color: str, alpha: float = 0.9) -> None:
        ax.add_collection3d(
            Poly3DCollection(faces, facecolors=color, edgecolors="k", linewidths=0.2, alpha=alpha)
        )

    add(base, "#4A769F")
    add(post, "#8AA3C0")
    add(inner, "#5B82D9")
    add(yaw_disk, "#1F77B4")
    add(panel, "#1F3A70")
    ax.set_title(
        (
            f"{mode.upper()} | h={h:5.1f} mm  yaw={yaw:+6.1f}°  "
            f"pitch={pitch:+5.1f}°  roll={roll:+5.1f}°"
        ),
        fontweight="bold",
        fontfamily="monospace",
    )


def run_viewer(
    mode: str,
    command_path: Path,
    output_path: Path | None,
    frames: int,
    fps: int,
    manual_pose: dict[str, float],
) -> None:
    geom = TwinGeometry()
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    live_records = _read_all_commands(command_path) if mode == "live" else []

    def _pose_for_frame(i: int) -> dict[str, float]:
        if mode == "auto":
            t = i / max(1, fps)
            return {
                "height": 65.0 + 65.0 * math.sin(2.0 * math.pi * t / 12.0),
                "yaw": geom.YAW_LIM * math.sin(2.0 * math.pi * t / 8.0),
                "pitch": geom.PITCH_LIM * math.sin(2.0 * math.pi * t / 5.0),
                "roll": geom.ROLL_LIM * math.cos(2.0 * math.pi * t / 4.0),
            }
        if mode == "manual":
            return dict(manual_pose)
        if output_path is None:
            latest = read_latest_command(command_path) or {}
        else:
            latest = live_records[min(i, len(live_records) - 1)] if live_records else {}
        return map_command_to_twin(latest)

    if output_path is None:
        for i in range(frames):
            pose = _pose_for_frame(i)
            render_frame(ax, pose, geom, mode=mode)
            plt.pause(1.0 / max(1, fps))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".png":
            pose = _pose_for_frame(max(0, frames - 1))
            render_frame(ax, pose, geom, mode=mode)
            fig.savefig(output_path, dpi=150)
        elif suffix == ".mp4":
            if animation.writers.is_available("ffmpeg"):
                def _animate(frame_idx: int) -> list[Any]:
                    pose = _pose_for_frame(frame_idx)
                    render_frame(ax, pose, geom, mode=mode)
                    return []

                ani = animation.FuncAnimation(
                    fig,
                    _animate,
                    frames=frames,
                    interval=1000.0 / max(1, fps),
                    blit=False,
                    repeat=False,
                )
                writer = animation.FFMpegWriter(fps=max(1, fps))
                ani.save(str(output_path), writer=writer)
            else:
                fallback_dir = output_path.with_suffix("")
                fallback_dir.mkdir(parents=True, exist_ok=True)
                for i in range(frames):
                    pose = _pose_for_frame(i)
                    render_frame(ax, pose, geom, mode=mode)
                    fig.savefig(fallback_dir / f"frame_{i + 1:04d}.png", dpi=150)
        else:
            pose = _pose_for_frame(max(0, frames - 1))
            render_frame(ax, pose, geom, mode=mode)
            fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AURA 3D sandbox viewer")
    p.add_argument("--mode", choices=("auto", "manual", "live"), default="live")
    p.add_argument("--commands", type=Path, default=Path("results/commands.jsonl"))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--height-mm", type=float, default=65.0)
    p.add_argument("--yaw", type=float, default=0.0)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--roll", type=float, default=0.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_viewer(
        mode=args.mode,
        command_path=args.commands,
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
