"""pi/daemon.py – Top-level main loop for the EDGE-AURA-MFP Pi daemon.

Reads aggregated sensor frames from a source (live UART or a binary
replay file), calibrates the raw values into engineering units, logs
everything locally, builds a SensorPacket matching the PINN-AURA-MFP
schema, and dispatches it to the orchestrator.

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
    --no-log        disable local SQLite+JSONL logging
    --validate      validate every packet against the schema (slower)
    --log-level     DEBUG | INFO | WARNING | ERROR  (default: INFO)

Typical uses
------------

Live:
    python -m pi.daemon

Offline replay (validation gate #2):
    python -m pi.daemon --input-mode file --input fixtures/sample_frames.bin \
                        --output-mode stdout --validate
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime
from typing import Optional

from pi.calibration    import Calibration
from pi.dispatch       import Dispatcher
from pi.logger         import Logger
from pi.packet_builder import build_packet, validate_packet
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


def _frame_to_packet(frame: dict, calibration: Calibration,
                     now_utc: Optional[datetime] = None) -> dict:
    flags = frame["fault_flags"]
    irradiance = calibration.pyranometer(frame["pyranometer_raw"], flags)
    tcs = calibration.thermocouple(frame["thermocouple_raw"], flags)
    speed, direction = calibration.anemometer(
        frame["anemometer_speed_x100"], frame["anemometer_dir_deg"], flags,
    )
    return build_packet(
        timestamp_ms=frame["timestamp_ms"],
        irradiance_w_m2=irradiance,
        thermocouples_c=tcs,
        wind_speed_m_s=speed,
        wind_direction_deg=direction,
        fault_flags=flags,
        image_path=None,           # Edge-Batch B wires the camera in
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
    try:
        if logger_cm is not None:
            logger_cm.open()
        with reader, dispatcher:
            for frame in reader:
                if stop[0]:
                    break
                if frame is None:
                    total_bad += 1
                    # Emit a degraded packet so downstream still sees the cadence.
                    degraded = build_packet(
                        timestamp_ms=0,
                        irradiance_w_m2=None,
                        thermocouples_c=[None] * 4,
                        wind_speed_m_s=None,
                        wind_direction_deg=None,
                        fault_flags=0x8000,     # persistent-fault bit
                        image_path=None,
                    )
                    if args.validate:
                        validate_packet(degraded)
                    dispatcher.send(degraded)
                    if logger_cm is not None:
                        logger_cm.write(degraded)
                    continue

                try:
                    pkt = _frame_to_packet(frame, calibration)
                    if args.validate:
                        validate_packet(pkt)
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
