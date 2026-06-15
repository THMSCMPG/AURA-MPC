"""pi/scripts/replay.py – Replay logged SensorPackets at a configurable speed.

Reads either:

* a JSONL file produced by ``pi.logger`` or ``pi.dispatch`` in file mode;
* a binary fixture file produced by the Pico (e.g. ``fixtures/sample_frames.bin``).

Each packet is re-dispatched through the same ``pi.dispatch.Dispatcher``
machinery the live daemon uses, optionally at a configurable speed
multiplier so long runs can be replayed quickly for debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pi.calibration    import Calibration
from pi.dispatch       import Dispatcher
from pi.packet_builder import build_packet
from pi.serial_reader  import SerialReader


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _iter_binary(path: Path, cal: Calibration):
    with SerialReader.from_file(str(path)) as reader:
        for frame in reader:
            if frame is None:
                continue
            flags = frame["fault_flags"]
            speed, direction = cal.anemometer(
                frame["anemometer_speed_x100"], frame["anemometer_dir_deg"], flags,
            )
            yield build_packet(
                timestamp_ms=frame["timestamp_ms"],
                irradiance_w_m2=cal.pyranometer(frame["pyranometer_raw"], flags),
                thermocouples_c=cal.thermocouple(frame["thermocouple_raw"], flags),
                wind_speed_m_s=speed,
                wind_direction_deg=direction,
                fault_flags=flags,
                image_path=None,
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Replay SensorPackets")
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--speed", type=float, default=1.0,
                   help="Replay-speed multiplier (1.0 = real time, "
                        "10.0 = 10× faster, 0 = as fast as possible).")
    p.add_argument("--output-mode", choices=["stdout", "socket", "file"], default="stdout")
    p.add_argument("--output", default=None)
    p.add_argument("--host",   default="127.0.0.1")
    p.add_argument("--port-out", type=int, default=9000)
    args = p.parse_args(argv)

    if not args.source.exists():
        p.error(f"source not found: {args.source}")

    if args.source.suffix == ".bin":
        packets = _iter_binary(args.source, Calibration())
    else:
        packets = _iter_jsonl(args.source)

    dispatcher = Dispatcher(
        mode=args.output_mode, host=args.host,
        port=args.port_out, path=args.output,
    )

    with dispatcher:
        prev_t: float | None = None
        for pkt in packets:
            # Respect the original inter-packet cadence when the packet
            # carries a ``t_s`` (Edge-Batch B PINN schema) or
            # ``timestamp_ms`` (legacy). ``--speed 0`` = no sleep;
            # ``--speed 1`` = real time; ``--speed 10`` = 10× faster.
            cur_t = _packet_time_seconds(pkt)
            if args.speed > 0 and prev_t is not None and cur_t is not None:
                dt = max(0.0, cur_t - prev_t) / args.speed
                if dt > 0:
                    time.sleep(dt)
            dispatcher.send(pkt)
            prev_t = cur_t if cur_t is not None else prev_t
    return 0


def _packet_time_seconds(pkt: dict) -> float | None:
    """Extract a monotonic timeline value from a replayed packet.

    Prefers the PINN ``t_s`` field (seconds) produced by Edge-Batch B;
    falls back to the legacy ``timestamp_ms`` / millisecond counter.
    """
    if "t_s" in pkt and pkt["t_s"] is not None:
        try:
            return float(pkt["t_s"])
        except (TypeError, ValueError):
            return None
    if "timestamp_ms" in pkt and pkt["timestamp_ms"] is not None:
        try:
            return float(pkt["timestamp_ms"]) / 1000.0
        except (TypeError, ValueError):
            return None
    return None


if __name__ == "__main__":
    sys.exit(main())
