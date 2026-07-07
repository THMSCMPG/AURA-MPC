"""pi/daemon.py – Top-level main loop for the EDGE-AURA-MFP Pi daemon.

Reads aggregated sensor frames from a source (live UART or a binary
replay file), calibrates the raw values into engineering units, logs
everything locally, builds a SensorPacket matching the **current**
PINN-AURA-MFP schema (``PINN_SENSOR_PACKET_SCHEMA`` /
``build_sensor_packet``), and dispatches it downstream.

.. note:: Schema fix (AURA-MPC wiring pass)
    This daemon previously built packets with the *deprecated*
    ``build_packet()`` / ``SENSOR_PACKET_SCHEMA`` — the schema every
    other current EDGE module (``pi.gateway``, ``pi.orchestrator_bridge``,
    ``pi.scripts.simulate``) had already moved off of. That meant the
    daemon's live output silently didn't match what the orchestrator /
    decision server actually expects (``G_poa``/``T_amb``/``WS``/``CC``/
    ``lat``/``lon`` instead of ``irradiance_w_m2``/``thermocouples_c``/...).
    This has been corrected to call ``build_sensor_packet`` /
    ``validate_sensor_packet`` instead. Two consequences worth knowing:

    1. The raw frame has no GPS/lat-lon (see ``pico/protocol/packet.py``).
       This is a **fixed installation** — ``--lat``/``--lon`` are now
       required daemon arguments (or set once in the systemd unit).
    2. The raw frame has 4 independent thermocouples but the new schema
       only carries one ``T_amb``. We report the mean of the valid
       (non-faulted) thermocouple readings as an ambient-temperature
       proxy — this is a documented approximation, not a dedicated
       ambient sensor; see ``_frame_to_pinn_packet`` below.

Command-line flags
------------------

    --input-mode    serial | file        (default: serial)
    --input         PATH                 (required when --input-mode=file)
    --port          /dev/serial0         (serial mode only)
    --baud          115200               (serial mode only)
    --output-mode   stdout | socket | file (default: stdout)
    --output        PATH                 (required when --output-mode=file)
    --host          127.0.0.1            (socket output mode)
    --port-out      9000                 (socket output mode)
    --lat           station latitude (required)
    --lon           station longitude (required)
    --no-log        disable local SQLite+JSONL logging
    --validate      validate every packet against the schema (slower)
    --log-level     DEBUG | INFO | WARNING | ERROR  (default: INFO)

Typical uses
------------

Live:
    python -m pi.daemon --lat 36.17 --lon -86.78

Offline replay (validation gate #2):
    python -m pi.daemon --input-mode file --input fixtures/sample_frames.bin \
                        --output-mode stdout --validate --lat 36.17 --lon -86.78
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime, timezone
from typing import Optional

from pi.calibration    import Calibration
from pi.dispatch       import Dispatcher
from pi.logger         import Logger
from pi.packet_builder import build_sensor_packet, validate_sensor_packet
from pi.serial_reader  import SerialReader

log = logging.getLogger("edge-aura.daemon")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDGE-AURA-MFP sensor daemon")
    p.add_argument("--input-mode", choices=["serial", "file", "stdin"], default="serial")
    p.add_argument("--input",      default=None,
                   help="Path to binary frame file (required when --input-mode=file)")
    p.add_argument("--port",       default="/dev/serial0")
    p.add_argument("--baud",       type=int, default=115_200)
    p.add_argument("--output-mode", choices=["stdout", "socket", "file"], default="stdout")
    p.add_argument("--output",     default=None,
                   help="Path to JSONL output file (required when --output-mode=file)")
    p.add_argument("--host",       default="127.0.0.1")
    p.add_argument("--port-out",   type=int, default=9000)
    p.add_argument("--lat",        type=float, default=None,
                   help="Station latitude (required — the edge has no GPS; this is a fixed install).")
    p.add_argument("--lon",        type=float, default=None,
                   help="Station longitude (required — see --lat).")
    p.add_argument("--no-log",     action="store_true",
                   help="Disable local SQLite+JSONL audit logging")
    p.add_argument("--validate",   action="store_true",
                   help="Validate every outgoing packet against the JSON schema")
    p.add_argument("--log-level",  default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    if args.input_mode == "file" and not args.input:
        p.error("--input is required when --input-mode=file")
    if args.output_mode == "file" and not args.output:
        p.error("--output is required when --output-mode=file")
    if args.lat is None or args.lon is None:
        p.error("--lat and --lon are required (the edge has no GPS; this is a fixed install)")
    return args


def _make_reader(args) -> SerialReader:
    if args.input_mode == "file":
        return SerialReader.from_file(args.input)
    if args.input_mode == "stdin":
        return SerialReader.from_stdin()
    return SerialReader(port=args.port, baud=args.baud)


def _make_dispatcher(args) -> Dispatcher:
    return Dispatcher(
        mode=args.output_mode,
        host=args.host,
        port=args.port_out,
        path=args.output,
    )


def _frame_to_pinn_packet(
    frame: dict,
    calibration: Calibration,
    *,
    lat: float,
    lon: float,
    t_s: float,
    now_utc: Optional[datetime] = None,
) -> dict:
    """Convert one raw pico frame into a current-schema ``SensorPacket``.

    Two approximations are made explicit here rather than silently
    baked into the calibration layer:

    * ``T_amb`` is the mean of the non-faulted thermocouple channels.
      The 4-thermocouple array measures panel-adjacent temperatures, not
      a dedicated shaded ambient-air sensor — treat this as a proxy
      until/unless a dedicated ambient probe is added to the BOM.
    * ``CC`` (cloud cover) is left ``None`` here; per the schema's own
      docstring it "is always produced by the orchestrator", not EDGE.
    """
    flags = frame["fault_flags"]
    irradiance = calibration.pyranometer(frame["pyranometer_raw"], flags)
    tcs = calibration.thermocouple(frame["thermocouple_raw"], flags)
    speed, _direction = calibration.anemometer(
        frame["anemometer_speed_x100"], frame["anemometer_dir_deg"], flags,
    )
    # NOTE: wind *direction* has no home in PINN_SENSOR_PACKET_SCHEMA (it
    # was dropped when the schema moved on from the deprecated
    # SensorPacket v1.0 shape). It is currently unused by the RK4TRAN
    # physics model either way (see src/RK4TRAN/evaluate_state.f90),
    # so this is a no-op today, but flagging it here in case that
    # changes and someone needs to re-add a wire for it.
    valid_tcs = [t for t in tcs if t is not None]
    t_amb = sum(valid_tcs) / len(valid_tcs) if valid_tcs else None

    return build_sensor_packet(
        t_s=t_s,
        G_poa=irradiance,
        T_amb=t_amb,
        WS=speed,
        lat=lat,
        lon=lon,
        fault_flags=flags,
        sky_image_path=None,       # Edge-Batch B wires the camera in
        now_utc=now_utc,
    )


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,           # keep stdout clean for pipe mode
    )

    calibration = Calibration()
    dispatcher = _make_dispatcher(args)
    reader = _make_reader(args)

    stop = [False]

    def _handle_signal(sig, _frame):
        log.info("signal %s – stopping", sig)
        stop[0] = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger_cm = Logger() if not args.no_log else None
    total_ok = 0
    total_bad = 0
    loop_start = datetime.now(timezone.utc)
    try:
        if logger_cm is not None:
            logger_cm.open()
        with reader, dispatcher:
            for frame in reader:
                if stop[0]:
                    break
                now_utc = datetime.now(timezone.utc)
                t_s = (now_utc - loop_start).total_seconds()
                if frame is None:
                    total_bad += 1
                    # Emit a degraded packet so downstream still sees the cadence.
                    degraded = build_sensor_packet(
                        t_s=t_s,
                        G_poa=None,
                        T_amb=None,
                        WS=None,
                        lat=args.lat,
                        lon=args.lon,
                        fault_flags=0x8000,     # persistent-fault bit
                        sky_image_path=None,
                        now_utc=now_utc,
                    )
                    if args.validate:
                        validate_sensor_packet(degraded)
                    dispatcher.send(degraded)
                    if logger_cm is not None:
                        logger_cm.write(degraded)
                    continue

                try:
                    pkt = _frame_to_pinn_packet(
                        frame, calibration,
                        lat=args.lat, lon=args.lon, t_s=t_s, now_utc=now_utc,
                    )
                    if args.validate:
                        validate_sensor_packet(pkt)
                except (ValueError, TypeError, KeyError) as exc:
                    log.error("calibration/validation error: %s", exc)
                    total_bad += 1
                    continue
                total_ok += 1
                dispatcher.send(pkt)
                if logger_cm is not None:
                    logger_cm.write(pkt)
    finally:
        if logger_cm is not None:
            logger_cm.close()
    log.info("daemon exited cleanly – %d packets ok, %d bad frames",
             total_ok, total_bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
