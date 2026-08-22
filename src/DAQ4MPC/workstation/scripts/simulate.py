"""workstation/scripts/simulate.py – Generate synthetic SensorPacket streams for integration testing.

Produces one of five named scenarios as newline-delimited JSON on stdout
(or file) so the full edge→PINN loop can be exercised without real
hardware.

STALE, FLAGGED NOT FIXED (2026-08-22): every scenario generator here still
simulates a real G_poa (irradiance) value via ``_diurnal_irradiance()`` --
the actual current Pico firmware never does this anymore (irradiance is a
manual calibration-time entry on the workstation, G_poa is always null on
the wire, see pico/json_builder.c). These generators are testing a
scenario the real hardware will never produce. Not fixed here -- this
file's main value (testing the live edge->PINN loop) depends on the
workstation-side decision_server.py integration, which is itself
deliberately being built last (see checklist). Revisit both together.

Usage::

    python -m workstation.scripts.simulate \\
        --scenario {null_images,partly_cloudy,fault_inject,deterministic,physics} \\
        --duration <seconds> \\
        [--cadence <seconds>]        # default 1.0  (0 = as fast as possible)
        [--seed <int>]               # default 0
        [--output-mode {stdout,file}]   # default stdout
        [--output <path>]            # required if output-mode=file
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

import numpy as np

from workstation.packet_builder import build_sensor_packet

# Default geographic location (Nashville, TN).
_DEFAULT_LAT: float = 36.17
_DEFAULT_LON: float = -86.78

# Fault flag constants – mirror pico/protocol/packet.py without importing pico.
_FAULT_PYRANOMETER: int = 0x0001  # pyranometer stuck / out of range
_FAULT_TC0: int = 0x0002          # thermocouple channel 0 open / fault

# Module-level interrupt flag; set by SIGINT handler, cleared at main() entry.
_INTERRUPTED: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ARG001
    """Signal handler – request a clean exit on Ctrl-C."""
    global _INTERRUPTED  # noqa: PLW0603
    _INTERRUPTED = True


def _diurnal_irradiance(t_h: float) -> float:
    """Clear-sky diurnal irradiance profile (cosine-weighted, noon peak ≈ 900 W/m²)."""
    return max(0.0, math.cos((t_h - 12.0) * math.pi / 12.0) * 900.0)


def _n_steps(duration: float, cadence: float) -> int:
    """Number of packets to emit for the given duration and cadence."""
    if cadence <= 0.0:
        # No real-time pacing: use 1 s nominal spacing to determine count.
        return max(1, int(duration))
    return max(1, round(duration / cadence))


# ──────────────────────────────────────────────────────────────────────────────
# Scenario generators – each yields SensorPacket dicts
# ──────────────────────────────────────────────────────────────────────────────

def _gen_null_images(
    rng: np.random.Generator,
    duration: float,
    cadence: float,
    start_utc: datetime,
) -> Iterator[dict]:
    """Clear-day diurnal profile with no sky image attached."""
    n = _n_steps(duration, cadence)
    t0_h = start_utc.hour + start_utc.minute / 60.0 + start_utc.second / 3600.0
    for i in range(n):
        t_h = t0_h + i / 3600.0
        T_amb = 20.0 + 5.0 * math.sin(math.pi * t_h / 12.0)
        yield build_sensor_packet(
            t_s=float(i),
            G_poa=_diurnal_irradiance(t_h),
            T_amb=T_amb,
            WS=float(rng.uniform(1.0, 5.0)),
            lat=_DEFAULT_LAT,
            lon=_DEFAULT_LON,
            fault_flags=0,
            CC=0.0,
            sky_image_path=None,
            now_utc=start_utc + timedelta(seconds=i),
        )


def _gen_partly_cloudy(
    rng: np.random.Generator,
    duration: float,
    cadence: float,
    start_utc: datetime,
) -> Iterator[dict]:
    """Diurnal profile with stochastic cloud cover (OU process) and synthetic sky images."""
    n = _n_steps(duration, cadence)
    t0_h = start_utc.hour + start_utc.minute / 60.0 + start_utc.second / 3600.0
    # Ornstein-Uhlenbeck parameters for cloud-cover process.
    theta, sigma, mu = 0.3, 0.2, 0.3
    cc: float = float(rng.uniform(0.0, 0.6))
    for i in range(n):
        # OU step: clamped to [0, 1].
        cc = cc + theta * (mu - cc) + sigma * float(rng.standard_normal())
        cc = max(0.0, min(1.0, cc))

        t_h = t0_h + i / 3600.0
        g_poa = _diurnal_irradiance(t_h) * (1.0 - 0.7 * cc)
        T_amb = 20.0 + 5.0 * math.sin(math.pi * t_h / 12.0)

        # Synthetic 32×32×3 uint8 sky image: clear = blue bias, overcast = grey.
        base_val = int(200 - 180 * cc)
        noise = rng.integers(-5, 6, size=(32, 32, 3), dtype=np.int16)
        img = np.clip(
            np.full((32, 32, 3), base_val, dtype=np.int16) + noise,
            0, 255,
        ).astype(np.uint8)
        if cc < 0.5:
            # Boost blue channel for clear skies.
            boost = int(60 * (1.0 - cc))
            img[:, :, 2] = np.clip(
                img[:, :, 2].astype(np.int16) + boost, 0, 255
            ).astype(np.uint8)

        pkt = build_sensor_packet(
            t_s=float(i),
            G_poa=g_poa,
            T_amb=T_amb,
            WS=float(rng.uniform(1.0, 5.0)),
            lat=_DEFAULT_LAT,
            lon=_DEFAULT_LON,
            fault_flags=0,
            CC=cc,
            sky_image_path=None,
            now_utc=start_utc + timedelta(seconds=i),
        )
        # Attach the synthetic sky image as a base64-encoded raw bytes field.
        # This is a simulation-only extension: 32×32×3 uint8, row-major, no
        # file I/O required.  The PINN image branch can decode with:
        #   np.frombuffer(base64.b64decode(pkt["sky_image"]), dtype=np.uint8).reshape(32,32,3)
        pkt["sky_image"] = base64.b64encode(img.tobytes()).decode("ascii")
        yield pkt


def _gen_fault_inject(
    rng: np.random.Generator,
    duration: float,
    cadence: float,
    start_utc: datetime,
) -> Iterator[dict]:
    """Clear-day for the first half; pyranometer + TC0 faults injected in the second half."""
    n = _n_steps(duration, cadence)
    half = n // 2
    t0_h = start_utc.hour + start_utc.minute / 60.0 + start_utc.second / 3600.0
    for i in range(n):
        t_h = t0_h + i / 3600.0
        T_amb = 20.0 + 5.0 * math.sin(math.pi * t_h / 12.0)
        # Inject faults in the second half only.
        fault_flags = 0 if i < half else (_FAULT_PYRANOMETER | _FAULT_TC0)
        yield build_sensor_packet(
            t_s=float(i),
            # Null irradiance when pyranometer is faulted.
            G_poa=None if fault_flags & _FAULT_PYRANOMETER else _diurnal_irradiance(t_h),
            T_amb=T_amb,
            WS=float(rng.uniform(1.0, 5.0)),
            lat=_DEFAULT_LAT,
            lon=_DEFAULT_LON,
            fault_flags=fault_flags,
            CC=0.0,
            sky_image_path=None,
            now_utc=start_utc + timedelta(seconds=i),
        )


def _gen_deterministic(
    rng: np.random.Generator,
    duration: float,
    cadence: float,
    start_utc: datetime,  # noqa: ARG001 – unused; fixed reference used instead
) -> Iterator[dict]:
    """Fixed-seed stream: two runs with the same seed produce byte-identical output."""
    n = _n_steps(duration, cadence)
    # Use a fixed reference timestamp so the output is independent of wall-clock
    # time and therefore fully reproducible.
    _REF_UTC = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i in range(n):
        yield build_sensor_packet(
            t_s=float(i),
            G_poa=float(rng.uniform(0.0, 900.0)),
            T_amb=float(rng.uniform(15.0, 35.0)),
            WS=float(rng.uniform(0.5, 8.0)),
            lat=_DEFAULT_LAT,
            lon=_DEFAULT_LON,
            fault_flags=0,
            CC=float(rng.uniform(0.0, 1.0)),
            sky_image_path=None,
            now_utc=_REF_UTC + timedelta(seconds=i),
        )


def _gen_physics(
    rng: np.random.Generator,
    duration: float,
    cadence: float,
    start_utc: datetime,
) -> Iterator[dict]:
    """Faiman-model panel temperature – physics-consistent (G_poa, T_amb, WS, CC, T_panel)."""
    n = _n_steps(duration, cadence)
    t0_h = start_utc.hour + start_utc.minute / 60.0 + start_utc.second / 3600.0
    for i in range(n):
        t_h = t0_h + i / 3600.0
        g_poa = _diurnal_irradiance(t_h)
        T_amb = 20.0 + 5.0 * math.sin(math.pi * t_h / 12.0)
        WS = max(0.1, float(rng.normal(3.0, 1.0)))
        cc = float(rng.uniform(0.0, 0.4))
        # Faiman panel thermal model.
        T_panel = T_amb + g_poa * (1.0 - cc) / (25.0 + 6.84 * WS)
        pkt = build_sensor_packet(
            t_s=float(i),
            G_poa=g_poa,
            T_amb=T_amb,
            WS=WS,
            lat=_DEFAULT_LAT,
            lon=_DEFAULT_LON,
            fault_flags=0,
            CC=cc,
            sky_image_path=None,
            now_utc=start_utc + timedelta(seconds=i),
        )
        # Attach physics-derived panel temperature for downstream regression.
        pkt["T_panel"] = round(T_panel, 4)
        yield pkt


_SCENARIO_GENERATORS: dict[str, Any] = {
    "null_images":   _gen_null_images,
    "partly_cloudy": _gen_partly_cloudy,
    "fault_inject":  _gen_fault_inject,
    "deterministic": _gen_deterministic,
    "physics":       _gen_physics,
}


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint-piping helper
# ──────────────────────────────────────────────────────────────────────────────

def _emit(pkt: dict[str, Any], *, output_mode: str, output_path) -> None:
    """Minimal stdout/file packet emitter, replacing the deleted
    workstation.dispatch.Dispatcher (which also supported a socket mode -- dropped
    here, doesn't match the new USB-serial architecture anyway; add back
    if genuinely needed later).
    """
    line = json.dumps(pkt, separators=(",", ":")) + "\n"
    if output_mode == "stdout":
        sys.stdout.write(line)
        sys.stdout.flush()
    elif output_mode == "file":
        with open(output_path, "a", encoding="utf-8") as fh:
            fh.write(line)
    else:
        raise ValueError(f"Unsupported output mode: {output_mode!r} (socket mode removed, see docstring)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    """Generate and dispatch synthetic SensorPacket streams."""
    global _INTERRUPTED  # noqa: PLW0603
    _INTERRUPTED = False
    signal.signal(signal.SIGINT, _handle_sigint)

    p = argparse.ArgumentParser(
        description="Generate synthetic SensorPacket streams for integration testing."
    )
    p.add_argument(
        "--scenario",
        required=True,
        choices=list(_SCENARIO_GENERATORS),
        help=(
            "Synthetic scenario to generate: "
            "null_images, partly_cloudy, fault_inject, deterministic, physics."
        ),
    )
    p.add_argument(
        "--duration", type=float, required=True,
        help="Simulation duration in seconds.",
    )
    p.add_argument(
        "--cadence", type=float, default=1.0,
        help="Packet inter-arrival interval in seconds (0 = as fast as possible).",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (default 0). Fixed seed → reproducible output.",
    )
    p.add_argument(
        "--output-mode", choices=["stdout", "file"], default="stdout",
    )
    p.add_argument(
        "--output", default=None,
        help="Output path (required when --output-mode=file).",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Target host for socket output mode.")
    p.add_argument("--port-out", type=int, default=9000,
                   help="Target port for socket output mode.")
    p.add_argument(
        "--checkpoint", default=None,
        help=(
            "PINN checkpoint path.  When set, spawn the orchestrator_bridge "
            "and pipe packets through it; OrchestrationCommands are printed on stdout."
        ),
    )
    p.add_argument(
        "--start-time", default=None,
        help="UTC start time as ISO-8601 (e.g. '2024-06-01T12:00:00Z'). "
             "Defaults to the current wall clock. Setting this makes runs "
             "fully reproducible and ensures scenarios that depend on solar "
             "elevation (partly_cloudy, fault_inject, null_images) "
             "exercise a daytime regime.",
    )
    args = p.parse_args(argv)

    if args.output_mode == "file" and not args.output:
        p.error("--output is required when --output-mode=file")

    rng = np.random.default_rng(args.seed)
    if args.start_time:
        start_utc = datetime.fromisoformat(args.start_time.replace("Z", "+00:00"))
        if start_utc.tzinfo is None:
            start_utc = start_utc.replace(tzinfo=timezone.utc)
    else:
        start_utc = datetime.now(timezone.utc)
    gen_fn = _SCENARIO_GENERATORS[args.scenario]
    packets = gen_fn(rng, args.duration, args.cadence, start_utc)

    if getattr(args, "checkpoint", None) is not None:
        sys.stderr.write(
            "simulate: --checkpoint (pipe into a live decision loop via "
            "orchestrator_bridge) was removed along with the Pi-hosted "
            "bridge architecture -- decision_server.py's raw_packet_to_pinn_packet() "
            "+ handle_packet() can be called in-process instead, but that "
            "integration hasn't been rebuilt here yet (deferred, workstation "
            "code is being built last -- see checklist). This flag is a no-op.\n"
        )
        return 1

    for pkt in packets:
        if _INTERRUPTED:
            break
        _emit(pkt, output_mode=args.output_mode, output_path=args.output)
        if args.cadence > 0.0:
            time.sleep(args.cadence)
    return 0


if __name__ == "__main__":
    sys.exit(main())
