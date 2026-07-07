"""decision_server.py – EDGE-facing TCP server closing the control loop.

This is the "workstation" half of::

    EDGE -> PINN -> RK4TRAN -> PINN -> DECISION -> EDGE

EDGE (``pi.gateway`` / ``pi.orchestrator_bridge``, running on the Pi) opens
a TCP connection here (``AURA_EDGE_BRIDGE_MODE=tcp_socket``), sends one
``PINN_SENSOR_PACKET_SCHEMA`` JSON line per sample window, and this server
replies with one actuator command JSON line per packet.

Per packet, this server:

1. Translates the EDGE packet into sandbox conditions
   (:func:`sandbox.edge_adapter.edge_packet_to_conditions`).
2. Injects those conditions into the running
   :class:`sandbox.runtime.ClosedLoopRuntime` episode
   (``PINN -> RK4TRAN -> PINN`` bias-corrected estimate happens inside
   :class:`sandbox.integration.SandboxPINNAgent`, invoked from
   :meth:`PanelEnv.step`).
3. Steps the trained policy (the DECISION layer) to get the next pose.
4. Converts that pose into an EDGE actuator command
   (:func:`sandbox.edge_adapter.pose_to_edge_command`) and replies.
5. Appends a full discrepancy/decision record — EDGE provenance plus the
   PINN, RK4TRAN, and decision outputs — to a JSONL log so drift and
   near-miss corrections are available for later PINN fine-tuning, not
   just for the live demo.

This deliberately does **not** replace ``pi.orchestrator_bridge`` — it
replaces what that bridge talks to. Point
``AURA_EDGE_BRIDGE_MODE=tcp_socket`` at this server's host:port instead of
the (currently nonexistent) ``python -m scripts.predict --mode live``
``subprocess_pipe`` target. See the module docstring in
``pi/orchestrator_bridge.py`` for the up-to-date transport guidance.

This server is intentionally single-connection / single-episode, mirroring
``workstation/inference_server.py``'s "one inference node" design note —
multi-EDGE-node fan-in is future work, not required for the first
end-to-end demo.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

from . import edge_adapter
from .matlab_bridge import _ensure_fortran_binary, _resolve_fortran_binary
from .runtime import ClosedLoopRuntime

log = logging.getLogger("decision_server")


class DecisionServer:
    """Blocking single-client TCP server wrapping :class:`ClosedLoopRuntime`."""

    def __init__(
        self,
        *,
        config_path: Path | str,
        pinn_checkpoint: Optional[Path | str] = None,
        policy_checkpoint: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        device: str = "cpu",
        host: str = "0.0.0.0",
        port: int = 8766,
        learning_enabled: bool = False,
        station_overrides: Optional[dict[str, float]] = None,
    ) -> None:
        cfg_path = Path(config_path).resolve()
        with cfg_path.open("r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        base_dir = cfg_path.parent.parent

        if pinn_checkpoint is None:
            pinn_checkpoint = base_dir / "outputs" / "pretrain" / "checkpoints" / "best_model.pt"
        if output_dir is None:
            output_dir = base_dir / "outputs" / "simulation"

        fortran_cfg = config.get("fortran", {})
        if "binary_path" in fortran_cfg:
            resolved = _resolve_fortran_binary(cfg_path, str(fortran_cfg["binary_path"]))
            fortran_cfg["binary_path"] = str(_ensure_fortran_binary(resolved))

        self.runtime = ClosedLoopRuntime(
            config=config,
            pinn_checkpoint=Path(pinn_checkpoint),
            policy_checkpoint=Path(policy_checkpoint) if policy_checkpoint else None,
            output_dir=Path(output_dir),
            device=device,
        )
        self.host = host
        self.port = int(port)
        self.learning_enabled = bool(learning_enabled)
        self.station_overrides = station_overrides or {}

        self._decision_log_path = Path(output_dir) / "edge_decision_log.jsonl"
        self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._decision_log_path.touch(exist_ok=True)

        # A live EDGE feed replaces the training-time "sample a random
        # episode" behaviour, so start from a neutral pose and let the
        # first real packet establish conditions.
        self.runtime.reset()

    # ------------------------------------------------------------------
    def handle_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Run one EDGE -> PINN -> RK4TRAN -> PINN -> DECISION cycle."""
        conditions_result = edge_adapter.edge_packet_to_conditions(
            packet, station_overrides=self.station_overrides
        )
        if conditions_result.degraded:
            log.warning(
                "decision_server: degraded packet — %s", "; ".join(conditions_result.notes)
            )

        self.runtime.inject_conditions(conditions_result.conditions)
        record = self.runtime.step(
            policy_mode="mean",
            learning_enabled=self.learning_enabled,
        )

        command = edge_adapter.pose_to_edge_command(
            record["pose"],
            decision_reason=record.get("decision_reason", ""),
            discrepancy=record.get("discrepancy"),
            validation=record.get("validation"),
        )

        self._append_decision_log(packet, conditions_result, record, command)
        return command

    def _append_decision_log(
        self,
        packet: dict[str, Any],
        conditions_result: edge_adapter.ConditionsResult,
        record: dict[str, Any],
        command: dict[str, Any],
    ) -> None:
        """Persist EDGE provenance + full decision context for later training.

        This is deliberately richer than the actuator-facing ``command``
        or the runtime's own ``closed_loop_trace.jsonl``: it ties a raw
        EDGE packet to the conditions actually used, the PINN estimate,
        the RK4TRAN validation (when performed), the discrepancy, and the
        resulting decision — the natural unit for a future fine-tuning
        dataset of "PINN was wrong here, by this much, under these
        conditions."
        """
        entry = {
            "edge_packet": packet,
            "conditions_used": conditions_result.conditions,
            "field_sources": conditions_result.field_sources,
            "degraded": conditions_result.degraded,
            "notes": conditions_result.notes,
            "pinn_prediction": record.get("pinn_prediction"),
            "rk4_prediction": record.get("rk4_prediction"),
            "discrepancy": record.get("discrepancy"),
            "decision_reason": record.get("decision_reason"),
            "command_sent": command,
        }
        with self._decision_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    def run(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            log.info("decision_server: listening on %s:%d", self.host, self.port)
            while True:
                conn, addr = srv.accept()
                log.info("decision_server: EDGE connection from %s", addr)
                self._handle_connection(conn)

    def _handle_connection(self, conn: socket.socket) -> None:
        buf = ""
        with conn:
            conn.settimeout(5.0)
            try:
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except socket.timeout:
                        continue
                    if not chunk:
                        break
                    buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        self._process_line(line, conn)
            except OSError as exc:
                log.debug("decision_server: connection closed — %s", exc)

    def _process_line(self, line: str, conn: socket.socket) -> None:
        try:
            packet = json.loads(line)
        except json.JSONDecodeError as exc:
            log.warning("decision_server: malformed EDGE packet JSON — %s", exc)
            return
        try:
            command = self.handle_packet(packet)
        except Exception as exc:  # noqa: BLE001 — never take the server down on one bad packet
            log.error("decision_server: handle_packet raised — %s", exc)
            return
        try:
            conn.sendall((json.dumps(command) + "\n").encode("utf-8"))
        except OSError as exc:
            log.warning("decision_server: send failed — %s", exc)


# ══════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="decision-server",
        description=(
            "EDGE-facing decision server: closes EDGE -> PINN -> RK4TRAN -> "
            "PINN -> DECISION -> EDGE. Point EDGE's orchestrator_bridge "
            "(AURA_EDGE_BRIDGE_MODE=tcp_socket) at this process's host:port."
        ),
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--config", required=True, metavar="PATH", help="sandbox.yaml path")
    p.add_argument("--checkpoint", default=None, metavar="PATH", help="PINN checkpoint (*.pt)")
    p.add_argument("--policy-checkpoint", default=None, metavar="PATH")
    p.add_argument("--output-dir", default=None, metavar="PATH")
    p.add_argument("--device", default="cpu")
    p.add_argument("--learning-enabled", action="store_true",
                    help="Update the policy online from live EDGE data (off by default in the field).")
    p.add_argument("--station-lat", type=float, default=None,
                    help="Override station latitude if not trusting the EDGE packet's lat.")
    p.add_argument("--station-lon", type=float, default=None)
    p.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    overrides: dict[str, float] = {}
    if args.station_lat is not None:
        overrides["lat"] = args.station_lat
    if args.station_lon is not None:
        overrides["lon"] = args.station_lon

    server = DecisionServer(
        config_path=args.config,
        pinn_checkpoint=args.checkpoint,
        policy_checkpoint=args.policy_checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        learning_enabled=args.learning_enabled,
        station_overrides=overrides,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
