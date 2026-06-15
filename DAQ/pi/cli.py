"""pi/cli.py – CLI entry point for the Pi Pico gateway.

Usage::

    python pi/cli.py --port /dev/ttyACM0 --baud 115200 \\
                     --window-size 60 --runner-path /usr/local/bin/simv2b

Or, after ``pip install -e .``:

    edge-gateway --port /dev/ttyACM0 --runner-path /usr/local/bin/simv2b
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="edge-gateway",
        description=(
            "Read PINN_SENSOR_PACKET_SCHEMA JSON from a Pi Pico over UART, "
            "validate each packet, buffer packets into windows, then "
            "trigger the AURA-MFP simv2b runner."
        ),
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port connected to the Pi Pico (default: /dev/ttyACM0).",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate (default: 115200).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=60,
        metavar="N",
        help="Number of valid packets per run window (default: 60).",
    )
    parser.add_argument(
        "--min-flush",
        type=int,
        default=10,
        metavar="N",
        help="Minimum packets needed to trigger a partial flush (default: 10).",
    )
    parser.add_argument(
        "--runner-path",
        default=None,
        metavar="PATH",
        help="Path to the simv2b runner binary (overrides AURA_MFP_SIMV2B_RUNNER env).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns an exit code (0 = success)."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    log = logging.getLogger("edge-aura.cli")

    import serial  # pyserial  # noqa: PLC0415

    from pi.gateway import Gateway  # noqa: PLC0415
    from pi.runner_bridge import call_simv2b_runner, packets_to_csv  # noqa: PLC0415

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    log.info("Opened %s @ %d baud", args.port, args.baud)

    gw = Gateway(ser, window_size=args.window_size, min_flush=args.min_flush)

    try:
        while True:
            packets = gw.ingest()
            if len(packets) < gw.min_flush:
                log.info(
                    "Serial source exhausted with %d packets (< min_flush=%d); stopping.",
                    len(packets),
                    gw.min_flush,
                )
                break
            csv_data = packets_to_csv(packets)
            metrics = call_simv2b_runner(csv_data, runner_path=args.runner_path)
            log.info("Window complete — metrics: %s", metrics)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
    finally:
        gw.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
