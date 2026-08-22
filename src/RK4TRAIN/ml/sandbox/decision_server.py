"""decision_server.py – EDGE-facing server closing the control loop.

This is the "workstation" half of::

    EDGE -> PINN (MPC candidate evaluation) -> DECISION -> EDGE

REWORKED this session (see checklist Section 1.12, runtime.py's module
docstring for the full history): the DECISION layer is no longer a trained
RL policy -- it's MPC candidate-evaluation using PINNSurrogate's fast
15-minute transient lookahead (D10). No weight updates happen here or
anywhere in the live path.

ARCHITECTURE UPDATE (this session): EDGE is now Pico-only (Pi 3B+ dropped)
talking USB serial directly to the workstation -- no separate Pi process,
no TCP hop, no network config. The Pico's C firmware (``DAQ4MPC/pico/``)
emits one ``PINN_SENSOR_PACKET_SCHEMA`` JSON line per sample over its USB
CDC port; this class's ``run_serial()`` reads that port directly and calls
``handle_packet()`` in-process -- no relay, no separate "gateway" process.
(TCP mode, ``run()``, is kept for anyone still testing without real
hardware attached, or a future non-USB EDGE variant -- not the primary
path anymore.)

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
4. Logs the recommendation and prints it for the operator to read and
   act on manually (no automated actuation -- confirmed this session,
   see ``handle_packet()``'s docstring).
5. Appends a full record — EDGE provenance plus every candidate's
   prediction/confidence, not just the winner's — to a JSONL log, which is
   the light predicted-vs-actual logging layer's input (tie a later
   real efficiency measurement back to a decision_id via
   ClosedLoopRuntime.record_outcome()).

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


def _import_calibration():
    """Lazily import DAQ4MPC.workstation.calibration.Calibration --
    cross-package import (this file lives under src/RK4TRAIN,
    calibration.py lives under src/DAQ4MPC/workstation), so the right
    directory needs to be on sys.path first. Done lazily (called from
    raw_packet_to_pinn_packet, not at module load time) so importing
    decision_server.py doesn't hard-fail for anyone not using the
    serial/raw-packet path."""
    daq4mpc_src = Path(__file__).resolve().parents[3] / "DAQ4MPC"
    if str(daq4mpc_src) not in sys.path:
        sys.path.insert(0, str(daq4mpc_src))
    from workstation.calibration import Calibration
    return Calibration


def raw_packet_to_pinn_packet(raw_packet: dict[str, Any], calibration=None) -> dict[str, Any]:
    """Translate one PICO_RAW_PACKET_SCHEMA packet (see pico/json_builder.h)
    into the existing PINN_SENSOR_PACKET_SCHEMA format that handle_packet()/
    edge_packet_to_conditions() already expect and are tested against.

    This is the ONE place calibration gets applied -- confirmed this
    session: calibration lives on the workstation, the Pico only emits
    raw ADC counts. Doing the translation here, before anything reaches
    handle_packet(), means the rest of the pipeline (edge_adapter.py,
    ClosedLoopRuntime, everything already built and tested) needed ZERO
    changes for this -- it still only ever sees calibrated, physical-unit
    packets in the same shape as before.
    """
    if calibration is None:
        Calibration = _import_calibration()
        calibration = Calibration()

    t_amb_c = calibration.t_amb(raw_packet["T_amb_raw"])
    ws = calibration.ws(raw_packet["WS_raw"])

    return {
        "timestamp": raw_packet["timestamp"],
        "t_s": raw_packet["t_s"],
        "G_poa": None,   # manual-entry, not sensed -- see checklist Section 1.12
        "T_amb": t_amb_c,
        "WS": ws,
        "CC": None,      # manual-entry, not sensed
        "lat": raw_packet["lat"],
        "lon": raw_packet["lon"],
        "sky_image_path": None,  # camera dropped
        "pose": None,
        "fault_flags": raw_packet["fault_flags"],
        "edge_version": raw_packet["edge_version"],
    }


def _edge_conditions_to_pinn_groups(
    conditions: dict[str, Any],
    *,
    irradiance: Optional[float] = None,
    cloud_cover: Optional[float] = None,
) -> dict[str, dict[str, float]]:
    """Translates edge_adapter.py's flat conditions dict
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

    irradiance/cloud_cover: passed in EXPLICITLY, not read from
    `conditions` -- edge_adapter.py's output deliberately never includes
    them (they're operator-supplied session constants, not sensed; a
    prior version derived them from the always-null G_poa/CC fields,
    which meant every post-calibration packet silently overwrote the
    operator's manually-entered irradiance back to 0.0 via
    inject_conditions()'s dict.update() semantics -- real bug, fixed
    2026-08-22). Pass None (the default) when you don't want these keys
    in the output at all -- e.g. handle_packet()'s per-packet call, where
    the goal is specifically to NOT touch the calibrated irradiance/
    cloud_cover values.
    """
    weather = {
        "T_amb": float(conditions["ambient_c"]) + 273.15,  # C -> K
        "wind_speed": float(conditions["wind_mps"]),
        "wind_dir": float(conditions["wind_dir"]),
        "humidity": float(conditions["humidity"]),
        "pressure": float(conditions["pressure"]),
    }
    if irradiance is not None:
        weather["irradiance"] = float(irradiance)
    if cloud_cover is not None:
        weather["cloud_cover"] = float(cloud_cover)
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
    def calibrate_from_packet(
        self,
        packet: dict[str, Any],
        target_position: dict[str, float],
        *,
        irradiance: float,
        cloud_cover: float,
    ) -> dict[str, Any]:
        """Must be called once, explicitly, before handle_packet() -- there
        is no more implicit "first packet establishes conditions" behavior.

        irradiance/cloud_cover are REQUIRED, explicit parameters -- they
        never come from the packet (G_poa/CC are always null on the wire,
        manual entry per the checklist). A previous version silently
        derived them from the packet instead (always 0.0, since always
        null) -- real bug, fixed 2026-08-22, see
        _edge_conditions_to_pinn_groups()'s docstring for the full story.
        """
        conditions_result = edge_adapter.edge_packet_to_conditions(packet, station_overrides=self.station_overrides)
        groups = _edge_conditions_to_pinn_groups(
            conditions_result.conditions, irradiance=irradiance, cloud_cover=cloud_cover
        )
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
        # (src/DAQ4MPC/workstation/packet_schema.py) -- there is no panel-temperature
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

    # ------------------------------------------------------------------
    def calibrate_interactive(self, ser, calibration=None) -> None:
        """Real calibration handshake for the serial workflow: read one
        raw packet off the given open serial connection, translate it
        (calibration applied here too, same as the main loop) for
        weather/location context, prompt the operator for a target
        position, confirm they've physically set it, and call
        runtime.calibrate(). Blocks on stdin input -- meant for an
        operator sitting at the terminal, not headless/unattended use.
        """
        if calibration is None:
            Calibration = _import_calibration()
            calibration = Calibration()

        log.info("Calibration: waiting for one packet from EDGE to read current weather/location context...")
        packet = None
        while packet is None:
            raw = ser.readline()
            if not raw:
                continue
            try:
                raw_packet = json.loads(raw.decode("utf-8", errors="replace").strip())
                packet = raw_packet_to_pinn_packet(raw_packet, calibration)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            except json.JSONDecodeError:
                continue

        print(f"\nGot a packet from EDGE: {packet}")
        print("Enter session weather (irradiance/cloud_cover are manual -- camera dropped, no live sensing):")

        # Sensible defaults pulled from the packet where available, computed
        # explicitly (not folded into the input() prompt) so it's obvious
        # and testable what each default actually is.
        packet_t_amb_c = packet.get("T_amb")
        default_t_amb_k = (packet_t_amb_c + 273.15) if packet_t_amb_c is not None else 293.15
        default_wind_speed = packet.get("WS") if packet.get("WS") is not None else 0.0

        def _prompt_float(label: str, default: float) -> float:
            raw = input(f"  {label} [{default}]: ").strip()
            return float(raw) if raw else float(default)

        weather = {
            "T_amb": _prompt_float("T_amb (K)", default_t_amb_k),
            "wind_speed": _prompt_float("wind_speed (m/s)", default_wind_speed),
            "wind_dir": _prompt_float("wind_dir (deg)", 0.0),
            "humidity": _prompt_float("humidity (0-1)", 0.5),
            "irradiance": _prompt_float("irradiance (W/m^2, manual)", 0.0),
            "cloud_cover": _prompt_float("cloud_cover (0-1, manual)", 0.0),
            "pressure": _prompt_float("pressure (Pa)", 101325.0),
        }
        location = {
            "lat": _prompt_float("lat", packet.get("lat", 0.0)),
            "lon": _prompt_float("lon", packet.get("lon", 0.0)),
            "elevation": _prompt_float("elevation (m)", 0.0),
        }
        print("Enter target position for the operator to physically set:")
        target_position = {
            "pitch": _prompt_float("target pitch (deg)", 0.0),
            "roll": _prompt_float("target roll (deg)", 0.0),
            "yaw": _prompt_float("target yaw (deg)", 0.0),
        }
        input("Physically set the panel to that position, then press Enter to confirm...")
        confirmed_position = dict(target_position)  # operator confirms by pressing Enter; adjust here if it differs

        record = self.runtime.calibrate(
            weather=weather, location=location,
            target_position=target_position, confirmed_position=confirmed_position,
            confirmed_readback=packet,
        )
        self._calibrated = True
        print(f"Calibrated. {record}\n")

    # ------------------------------------------------------------------
    def run_serial(self, port: str, baud: int = 115200) -> None:
        """Primary Pico-only entry point: read the Pico's USB CDC serial
        port directly and call handle_packet() in-process. No relay, no
        separate gateway process, no reply written back -- the Pico only
        ever emits (manual actuation means there's nothing for it to
        receive), and the recommendation is already logged/printed by
        handle_packet() itself for the operator to read.

        Requires calibrate_from_packet() to have been called first (same
        as the TCP path) -- deliberately does NOT auto-calibrate from the
        first packet with a placeholder position. Calibration exists
        specifically so the operator confirms a KNOWN position before any
        recommendation is trusted; silently calibrating against whatever
        pose happened to be sensed first would defeat that entirely. If
        you haven't calibrated yet, this logs a clear error per packet
        and keeps waiting rather than crashing or guessing.
        """
        import serial  # pyserial -- imported lazily so TCP-only usage has no hard dep
        Calibration = _import_calibration()
        calibration = Calibration()  # one instance, reused for every packet -- loads calibration/*.json once

        log.info("decision_server: opening serial port %s at %d baud", port, baud)
        with serial.Serial(port, baud, timeout=5.0) as ser:
            if not self._calibrated:
                self.calibrate_interactive(ser, calibration)

            while True:
                raw = ser.readline()
                if not raw:
                    continue  # timeout with no data -- keep waiting, Pico may just be between samples
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception as exc:  # noqa: BLE001
                    log.warning("decision_server: undecodable serial line — %s", exc)
                    continue
                if not line:
                    continue
                try:
                    raw_packet = json.loads(line)
                except json.JSONDecodeError as exc:
                    log.warning("decision_server: malformed EDGE packet JSON — %s (%r)", exc, line[:200])
                    continue

                try:
                    packet = raw_packet_to_pinn_packet(raw_packet, calibration)
                except (KeyError, TypeError) as exc:
                    log.warning("decision_server: raw packet missing expected fields — %s (%r)", exc, raw_packet)
                    continue

                try:
                    self.handle_packet(packet)
                except RuntimeError as exc:
                    log.error("decision_server: %s -- run calibrate_from_packet() first", exc)
                except Exception as exc:  # noqa: BLE001 — never take the server down on one bad packet
                    log.error("decision_server: handle_packet raised — %s", exc)


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
    p.add_argument("--transport", choices=["serial", "tcp"], default="serial",
                    help="serial: read the Pico's USB CDC port directly (Pico-only design, primary path). "
                         "tcp: legacy socket listener, for testing without hardware attached.")
    p.add_argument("--serial-port", default="/dev/ttyACM0",
                    help="Pico's USB CDC serial device (serial transport only). "
                         "Typically /dev/ttyACM0 on Linux, COM<N> on Windows.")
    p.add_argument("--serial-baud", type=int, default=115200)
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
    if args.transport == "serial":
        server.run_serial(args.serial_port, args.serial_baud)
    else:
        server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
