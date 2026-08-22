"""decision_server.py – EDGE-facing TCP server closing the control loop.

This is the "workstation" half of::

    EDGE -> PINN (MPC candidate evaluation) -> DECISION -> EDGE

REWORKED this session (see checklist Section 1.12, runtime.py's module
docstring for the full history): the DECISION layer is no longer a trained
RL policy -- it's MPC candidate-evaluation using PINNSurrogate's fast
15-minute transient lookahead (D10). No weight updates happen here or
anywhere in the live path.

EDGE (``pi.gateway`` / ``pi.orchestrator_bridge``, running on the Pi) opens
a TCP connection here (``AURA_EDGE_BRIDGE_MODE=tcp_socket``), sends one
``PINN_SENSOR_PACKET_SCHEMA`` JSON line per sample window, and this server
replies with one pose recommendation JSON line per packet.

Per packet (AFTER an explicit calibrate_from_packet() call -- see
DecisionServer's docstring, there's no more implicit "first packet
establishes conditions"), this server:

1. Translates the EDGE packet into sandbox conditions
   (:func:`sandbox.edge_adapter.edge_packet_to_conditions`).
2. Injects live-sensed fields (e.g. T_amb, wind) into the session's
   conditions -- weather fields the camera/PSO estimator used to provide
   (irradiance, cloud_cover) are now manual entries fixed at calibration,
   not derived per-packet.
3. Runs one MPC decision cycle (:meth:`sandbox.runtime.ClosedLoopRuntime.recommend`):
   propose candidate orientations, evaluate each with the PINN's fast
   transient lookahead, argmax by predicted cooling (D6).
4. Converts the chosen candidate into an EDGE-facing message
   (:func:`sandbox.edge_adapter.pose_to_edge_command`) and replies.
5. Appends a full record — EDGE provenance plus every candidate's
   prediction/confidence, not just the winner's — to a JSONL log, which is
   the light predicted-vs-actual logging layer's input (tie a later
   real efficiency measurement back to a decision_id via
   ClosedLoopRuntime.record_outcome()).

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


def _edge_conditions_to_pinn_groups(conditions: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Translates edge_adapter.py's flat EpisodeConditions-style dict
    (ambient_c, wind_mps, alt, ... -- see that module's docstring) into the
    grouped weather/location/time dicts PINNValidator.predict() expects
    (T_amb, wind_speed, elevation, ...).

    WHY THIS EXISTS: edge_adapter.py was built against the now-deleted
    PanelEnv/EpisodeConditions, which apparently did this translation
    silently inside PanelEnv.set_conditions(). With PanelEnv gone (see
    checklist Section 1.12), this decision_server calls PINNValidator
    directly, and the two sides use genuinely different field names AND at
    least one different unit -- confirmed via edge_adapter.py's own default
    value: ambient_c defaults to 25.0 with a comment "assuming 25 C",
    i.e. Celsius, while RK4TRAN's T_amb is Kelvin throughout (T_STC=298.15
    K, validated repeatedly this session). Found by actually running a
    mocked packet through the new code path, not by inspection -- it threw
    a real KeyError before this existed.
    """
    weather = {
        "T_amb": float(conditions["ambient_c"]) + 273.15,  # C -> K
        "wind_speed": float(conditions["wind_mps"]),
        "wind_dir": float(conditions["wind_dir"]),
        "humidity": float(conditions["humidity"]),
        "irradiance": float(conditions["irradiance"]),
        "cloud_cover": float(conditions["cloud_cover"]),
        "pressure": float(conditions["pressure"]),
    }
    location = {
        "lon": float(conditions["lon"]),
        "lat": float(conditions["lat"]),
        "elevation": float(conditions["alt"]),
    }
    time_components = {
        "minute": float(conditions.get("minute", 0.0)),
        "hour": float(conditions.get("hour", 12.0)),
        "day_of_year": float(conditions.get("day_of_year", 180.0)),
        "month": float(conditions.get("month", 6.0)),
        "year": float(conditions.get("year", 2026.0)),
    }
    return {"weather": weather, "location": location, "time": time_components}


class DecisionServer:
    """Blocking single-client TCP server wrapping :class:`ClosedLoopRuntime`.

    REWORKED (see checklist Section 1.12, runtime.py's module docstring for
    the full history): no more policy weight updates, no more
    ``learning_enabled``/``policy_checkpoint`` -- the DECISION layer is now
    MPC candidate-evaluation via the PINN's fast transient lookahead, not a
    trained action policy.

    CONFIRMED: manual actuation. This does NOT send a move command to EDGE
    (resolved this session -- previously flagged as open, now fixed).
    handle_packet() logs/prints the recommendation and returns a plain
    acknowledgment; the operator moves the panel by hand and (later) calls
    ClosedLoopRuntime.record_outcome() once new sensor data confirms the
    real effect. edge_decision_log.jsonl captures every candidate's
    prediction/confidence, not just the winner's, for that analysis.

    Requires an EXPLICIT calibration packet/call before any recommendations
    are served -- there is no more "start from a neutral pose and let the
    first real packet establish conditions" (that was PanelEnv.reset()'s
    old behavior). See ``calibrate_from_packet`` below.
    """

    def __init__(
        self,
        *,
        config_path: Path | str,
        pinn_checkpoint: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        device: str = "cpu",
        host: str = "0.0.0.0",
        port: int = 8766,
        n_candidates: int = 12,
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
        rk4_binary = None
        if "binary_path" in fortran_cfg:
            resolved = _resolve_fortran_binary(cfg_path, str(fortran_cfg["binary_path"]))
            rk4_binary = _ensure_fortran_binary(resolved)

        self.runtime = ClosedLoopRuntime(
            pinn_checkpoint=Path(pinn_checkpoint),
            output_dir=Path(output_dir),
            rk4_binary=rk4_binary,
            device=device,
            n_candidates=n_candidates,
        )
        self.host = host
        self.port = int(port)
        self.station_overrides = station_overrides or {}
        self._calibrated = False

        self._decision_log_path = Path(output_dir) / "edge_decision_log.jsonl"
        self._decision_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._decision_log_path.touch(exist_ok=True)

    # ------------------------------------------------------------------
    def calibrate_from_packet(self, packet: dict[str, Any], target_position: dict[str, float]) -> dict[str, Any]:
        """Must be called once, explicitly, before handle_packet() -- there
        is no more implicit "first packet establishes conditions" behavior.
        Per the manual calibration workflow: weather (including irradiance/
        cloud_cover, now manual set-value entries -- camera dropped, PSO
        irradiance estimator cut this session) comes from the caller, not
        derived from the packet.
        """
        conditions_result = edge_adapter.edge_packet_to_conditions(packet, station_overrides=self.station_overrides)
        groups = _edge_conditions_to_pinn_groups(conditions_result.conditions)
        record = self.runtime.calibrate(
            weather=groups["weather"],
            location=groups["location"],
            target_position=target_position,
            confirmed_position=target_position,  # caller should override once EDGE confirms actual position
            confirmed_readback=conditions_result.conditions,
        )
        self._calibrated = True
        return record

    # ------------------------------------------------------------------
    def handle_packet(self, packet: dict[str, Any]) -> dict[str, Any]:
        """Run one EDGE -> PINN -> DECISION cycle.

        MANUAL WORKFLOW (confirmed): this does NOT send an actuator command
        to EDGE. It computes and logs the recommendation, prints it
        prominently for the operator to read, and sends EDGE a plain
        acknowledgment -- not a move instruction. The operator moves the
        panel by hand; there's no automated actuator to command yet.
        """
        if not self._calibrated:
            raise RuntimeError("calibrate_from_packet() must be called before handle_packet()")

        conditions_result = edge_adapter.edge_packet_to_conditions(
            packet, station_overrides=self.station_overrides
        )
        if conditions_result.degraded:
            log.warning(
                "decision_server: degraded packet — %s", "; ".join(conditions_result.notes)
            )

        groups = _edge_conditions_to_pinn_groups(conditions_result.conditions)
        self.runtime.inject_conditions(weather=groups["weather"], location=groups["location"])

        # T_panel_current: checked the REAL packet schema this session
        # (src/DAQ4MPC/pi/packet_schema.py) -- there is no panel-temperature
        # field anywhere in it (schema is timestamp_iso, G_poa, T_amb, WS,
        # CC, lat, lon, azimuth, tilt, height, fault_flags, sky_image_path,
        # pose, edge_version). This isn't a naming mismatch to fix -- the
        # hardware doesn't sense panel temperature at all right now. Two
        # real ways to close this gap, neither implemented here:
        #   (a) add a panel-temperature sensor to the DAQ hardware (this is
        #       exactly the kind of thing worth deciding alongside the
        #       Pi3B+-vs-Pico hardware simplification you're weighing)
        #   (b) track the panel's actual current pose across decisions
        #       session-side and have the runtime compute its OWN
        #       steady-state estimate as a proxy (needs new state ClosedLoopRuntime
        #       doesn't currently keep -- not built, would be real scope)
        # Interim fallback used here: T_amb. Physically defensible as a
        # baseline (panel temperature tracks ambient absent other info) but
        # genuinely just a placeholder -- flagged plainly, not hidden.
        T_panel_current = groups["weather"]["T_amb"]
        time_components = groups["time"]
        record = self.runtime.recommend(T_panel_current=T_panel_current, time_components=time_components)

        chosen = record["chosen"]
        log.info(
            "RECOMMENDATION [%s]: move panel to pitch=%.1f deg, yaw=%.1f deg, roll=%.1f deg "
            "(predicted cooling=%.2f K, argmax over %d candidates)",
            record["decision_id"], chosen["pitch"], chosen["yaw"], chosen["roll"],
            chosen["predicted_cooling"], len(record["candidates"]),
        )

        # Acknowledgment only -- NOT edge_adapter.pose_to_edge_command().
        # That function builds an actuator command; per the confirmed manual
        # workflow, EDGE never receives a move instruction from this server.
        # The recommendation is logged/printed here for the operator to act
        # on by hand, and captured in full (all candidates + confidences,
        # not just the winner) in edge_decision_log.jsonl for the light
        # predicted-vs-actual analysis (tie a later record_outcome() call
        # back to decision_id).
        response = {
            "schema_version": "recommendation-1.0",
            "decision_id": record["decision_id"],
            "recommended_pose": {"pitch_deg": chosen["pitch"], "yaw_deg": chosen["yaw"], "roll_deg": chosen["roll"]},
            "predicted_cooling_K": chosen["predicted_cooling"],
            "note": "recommendation only -- no automated actuation, move the panel manually",
        }

        self._append_decision_log(packet, conditions_result, record, response)
        return response

    def _append_decision_log(
        self,
        packet: dict[str, Any],
        conditions_result: edge_adapter.ConditionsResult,
        record: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Persist EDGE provenance + full decision context, including every
        candidate's confidence (not just the winner) -- this IS the light
        predicted-vs-actual logging layer's input data, tie decision_id
        here to a later record_outcome() call once real efficiency data
        comes in.
        """
        entry = {
            "edge_packet": packet,
            "conditions_used": conditions_result.conditions,
            "field_sources": conditions_result.field_sources,
            "degraded": conditions_result.degraded,
            "notes": conditions_result.notes,
            "decision_id": record.get("decision_id"),
            "candidates": record.get("candidates"),
            "chosen": record.get("chosen"),
            "rk4_steady_state_check": record.get("rk4_steady_state_check"),
            "discrepancy": record.get("discrepancy"),
            "response_sent": response,
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
            response = self.handle_packet(packet)
        except Exception as exc:  # noqa: BLE001 — never take the server down on one bad packet
            log.error("decision_server: handle_packet raised — %s", exc)
            return
        try:
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
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
    p.add_argument("--output-dir", default=None, metavar="PATH")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-candidates", type=int, default=12,
                    help="Candidate orientations evaluated per decision (MPC, not a learned policy -- see D5).")
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
        output_dir=args.output_dir,
        device=args.device,
        host=args.host,
        port=args.port,
        n_candidates=args.n_candidates,
        station_overrides=overrides,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
