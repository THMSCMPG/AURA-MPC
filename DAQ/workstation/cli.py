"""workstation/cli.py – CLI entry point for the AURA workstation inference server.

Usage::

    python workstation/cli.py \\
        --host 0.0.0.0 --port 8765 \\
        --aura-root /path/to/AURA-MFP \\
        --pinn-root /path/to/PINN-AURA-MFP \\
        --checkpoint /path/to/checkpoint.pt \\
        --optimiser pso \\
        --n-particles 50

Or, after ``pip install -e .``::

    workstation-server --host 0.0.0.0 --port 8765 ...
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="workstation-server",
        description=(
            "AURA workstation inference server.  "
            "Receives PINN_SENSOR_PACKET_SCHEMA JSON from the Pi 3B+ gateway, "
            "routes to the appropriate Fortran solver tier, runs PSO/BO "
            "optimisation, and returns an OptimalConfigCommand JSON."
        ),
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address for the TCP server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="TCP port to listen on (default: 8765).",
    )
    parser.add_argument(
        "--aura-root",
        default=None,
        metavar="PATH",
        help="Path to the AURA-MFP repository root.",
    )
    parser.add_argument(
        "--pinn-root",
        default=None,
        metavar="PATH",
        help="Path to the PINN-AURA-MFP repository root.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        help="Path to the trained PINN checkpoint file (*.pt).",
    )
    parser.add_argument(
        "--optimiser",
        default="pso",
        choices=["pso", "bo"],
        help="Optimisation algorithm: pso (default) or bo.",
    )
    parser.add_argument(
        "--n-particles",
        type=int,
        default=50,
        metavar="N",
        help="PSO swarm size (default: 50).",
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

    log = logging.getLogger("workstation.cli")
    log.info(
        "Starting inference server on %s:%d (optimiser=%s, n_particles=%d)",
        args.host,
        args.port,
        args.optimiser,
        args.n_particles,
    )

    from workstation.inference_server import InferenceServer  # noqa: PLC0415

    server = InferenceServer(
        host=args.host,
        port=args.port,
        aura_root=args.aura_root,
        pinn_root=args.pinn_root,
        checkpoint=args.checkpoint,
        optimiser=args.optimiser,
        n_particles=args.n_particles,
    )
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
