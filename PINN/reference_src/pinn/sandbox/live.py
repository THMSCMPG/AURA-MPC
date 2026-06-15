"""Live-mode bridge for orchestrator-backed command generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

import torch
from torch import Tensor

from ...config import AppConfig
from ..data import SensorPacket
from ..orchestrator import RealTimeOrchestrator
from .environment import PanelEnv
from .training import PolicyNet


class SensorStream(Protocol):
    """Pluggable stream interface.

    The test wrapper uses :class:`JsonlSensorStream` (mock DAQ). A future hardware
    wrapper can implement this protocol and replace the source without changing
    policy execution or command logging.
    """

    def __iter__(self) -> "SensorStream":
        return self

    def __next__(self) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class JsonlSensorStream:
    """Simple JSONL-backed stream used by live_test mock demos."""

    path: Path
    _fh: TextIO | None = None

    def __iter__(self) -> "JsonlSensorStream":
        self._fh = self.path.open("r", encoding="utf-8")
        return self

    def __next__(self) -> dict[str, Any]:
        if self._fh is None:
            raise StopIteration
        while True:
            line = self._fh.readline()
            if line == "":
                self._fh.close()
                self._fh = None
                raise StopIteration
            line = line.strip()
            if line:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("sensor stream line must decode to a JSON object")
                return parsed


class _LowUncertaintyDemoModel(torch.nn.Module):
    """Tiny stand-in model for demos when no PINN checkpoint is provided."""

    def __init__(self, n_routes: int) -> None:
        super().__init__()
        self._n_routes = max(1, int(n_routes))

    def predict_with_uncertainty(self, numeric: Tensor, image: Tensor) -> dict[str, Tensor]:
        batch = numeric.shape[0]
        probs = torch.full((batch, self._n_routes), 0.01, dtype=torch.float32)
        preferred_idx = min(self._n_routes - 1, 3)
        probs[:, preferred_idx] = 0.96
        probs = probs / probs.sum(dim=1, keepdim=True)
        t_hat = numeric[:, 2:3].to(dtype=torch.float32) + 273.15
        return {
            "T_hat": t_hat,
            "route_probs": probs,
            "route_uncertainty": 1.0 - probs.max(dim=-1).values,
        }

    def eval(self) -> "_LowUncertaintyDemoModel":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "_LowUncertaintyDemoModel":
        return self


def _parse_timestamp(record: dict[str, Any]) -> datetime:
    raw = record.get("timestamp") or record.get("timestamp_utc")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    ts_ms = record.get("ts_ms", record.get("timestamp_ms"))
    if isinstance(ts_ms, (int, float)):
        return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
    return datetime.now(tz=timezone.utc)


def sensor_packet_from_record(record: dict[str, Any]) -> SensorPacket:
    """Normalize stream records to :class:`SensorPacket`.

    Supports direct SensorPacket-shaped JSON and the EDGE fixture schema.
    """
    ts = _parse_timestamp(record)
    t_s_raw = record.get("t_s")
    if t_s_raw is None:
        ts_ms = record.get("ts_ms", record.get("timestamp_ms"))
        if isinstance(ts_ms, (int, float)):
            t_s_raw = float(ts_ms) / 1000.0
        else:
            t_s_raw = (
                ts.hour * 3600
                + ts.minute * 60
                + ts.second
                + ts.microsecond / 1_000_000.0
            )
    t_s_seconds = float(t_s_raw) % 86400.0
    g_poa_raw = record.get("G_poa")
    if g_poa_raw is None:
        pyr_raw = record.get("pyranometer_raw")
        g_poa_raw = float(pyr_raw) * 0.0192 if isinstance(pyr_raw, (int, float)) else 0.0
    t_amb_raw = record.get("T_amb")
    if t_amb_raw is None:
        t_amb_raw = record.get("bme280_temp", record.get("thermistor_0", 25.0))
    ws_raw = record.get("WS")
    if ws_raw is None:
        ws_raw = record.get("anemometer_hz", 0.0)
    cc_raw = record.get("CC", record.get("cloud_cover", 0.0))
    pose_raw = record.get("pose")
    pose: dict[str, float] | None = None
    if isinstance(pose_raw, dict):
        pose = {
            k: float(v)
            for k, v in pose_raw.items()
            if k in {"pitch", "yaw", "roll", "z"} and isinstance(v, (int, float))
        }
    return SensorPacket(
        timestamp=ts,
        t_s=t_s_seconds,
        G_poa=float(g_poa_raw),
        T_amb=float(t_amb_raw),
        WS=float(ws_raw),
        CC=float(cc_raw),
        lat=float(record.get("lat", 36.53)),
        lon=float(record.get("lon", -87.36)),
        pose=pose,
        fault_flags=int(record.get("fault_flags", 0)),
        edge_version=str(record.get("edge_version", record.get("schema_version", "unknown"))),
        sky_image_path=(str(record["sky_image_path"]) if "sky_image_path" in record else None),
    )


def _policy_seed_pose(cfg: AppConfig, checkpoint_path: Path) -> dict[str, float]:
    """Use sandbox policy checkpoint to seed orchestrator starting pose."""
    env = PanelEnv(cfg)
    obs, _ = env.reset(seed=cfg.sandbox.seed)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("policy_state", {})
    if not isinstance(state, dict):
        raise ValueError("checkpoint missing policy_state")
    policy = PolicyNet(env.observation_dim, cfg.sandbox.policy_hidden_dim, env.action_dim)
    policy.load_state_dict(state)
    policy.eval()
    with torch.no_grad():
        action = policy(torch.from_numpy(obs).float().unsqueeze(0)).squeeze(0).cpu().numpy()
    _, _, _, _, info = env.step(action)
    pose = info.get("pose", {})
    if not isinstance(pose, dict):
        raise ValueError("sandbox policy seed did not produce a valid pose")
    return {
        "pitch": float(pose.get("pitch", 0.0)),
        "yaw": float(pose.get("yaw", 0.0)),
        "roll": float(pose.get("roll", 0.0)),
        "z": float(pose.get("z", 1.0)),
    }


def run_live_stream(
    cfg: AppConfig,
    orchestrator_checkpoint: Path | None,
    policy_checkpoint: Path | None,
    stream: SensorStream,
    output_jsonl: Path,
    aura_mfp_root: Path | None = None,
    *,
    realtime: bool = False,
) -> int:
    """Consume stream records and append orchestrator commands to JSONL."""
    model: Any | None = None
    model_path = orchestrator_checkpoint
    if model_path is None:
        model = _LowUncertaintyDemoModel(n_routes=len(cfg.model.route_labels))
        model_path = policy_checkpoint if policy_checkpoint is not None else Path("unused.pt")
    orchestrator = RealTimeOrchestrator(
        model_path=model_path,
        config=cfg,
        aura_mfp_root=aura_mfp_root,
        model=model,
    )

    n = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with orchestrator, output_jsonl.open("a", encoding="utf-8") as fh:
        if policy_checkpoint is not None and policy_checkpoint.exists():
            # Integration point for phase-2 sandbox policy: use the policy to
            # seed the orchestrator's initial control state, but keep runtime
            # decisioning inside RealTimeOrchestrator (watchdog/routing/control).
            orchestrator._current_pose = _policy_seed_pose(cfg, policy_checkpoint)  # noqa: SLF001
        for record in stream:
            # TODO(phase-3): replace JsonlSensorStream with a real DAQ driver
            # that implements SensorStream and yields SensorPacket-compatible records.
            packet = sensor_packet_from_record(record)
            command = orchestrator.step(packet)
            fh.write(command.to_json() + "\n")
            fh.flush()
            n += 1
            if realtime:
                time.sleep(float(cfg.contract.decision_cadence_s))
    return n
