"""pi/scripts/health_check.py – Verify serial link and sensor sanity.

Opens the serial port (or a replay file, or runs a mock pass), collects
packets through the full calibration pipeline, and prints per-field
pass/fail against expected operating ranges. Exits 0 on pass, 1 on
failure.

.. note:: Schema fix (AURA-MPC wiring pass)
    ``_collect`` previously round-tripped every reading through the
    deprecated ``build_packet`` just to pull the calibrated values back
    out again — coupling a per-channel sanity check to a specific wire
    schema for no reason. It now calls ``Calibration`` directly. The
    wire-schema check itself (item 5 of ``--mock``) already validates
    against the *current* schema via ``build_sensor_packet`` /
    ``validate_sensor_packet`` and is unaffected by this.

Usage
-----
    python -m pi.scripts.health_check [--port /dev/serial0] [--baud 115200] [--count 10]
    python -m pi.scripts.health_check --input-mode file --input fixtures/sample_frames.bin
    python -m pi.scripts.health_check --mock         # CI / pre-flight, no hardware
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

from pi.calibration    import Calibration
from pi.serial_reader  import SerialReader

# (field, lo, hi)
_RANGES = [
    ("irradiance_w_m2",    -10.0, 2000.0),
    ("tc0_c",              -40.0,  200.0),
    ("tc1_c",              -40.0,  200.0),
    ("tc2_c",              -40.0,  200.0),
    ("tc3_c",              -40.0,  200.0),
    ("wind_speed_m_s",       0.0,   60.0),
    ("wind_direction_deg",   0.0,  360.0),
]


def _collect(reader: SerialReader, cal: Calibration, count: int) -> dict:
    samples: dict = {name: [] for name, _, _ in _RANGES}
    seen = 0
    for frame in reader:
        if frame is None:
            continue
        flags = frame["fault_flags"]
        irradiance = cal.pyranometer(frame["pyranometer_raw"], flags)
        thermocouples = cal.thermocouple(frame["thermocouple_raw"], flags)
        wind_speed, wind_direction = cal.anemometer(
            frame["anemometer_speed_x100"], frame["anemometer_dir_deg"], flags,
        )
        if irradiance is not None:
            samples["irradiance_w_m2"].append(irradiance)
        for i, v in enumerate(thermocouples):
            if v is not None:
                samples[f"tc{i}_c"].append(v)
        if wind_speed is not None:
            samples["wind_speed_m_s"].append(wind_speed)
        if wind_direction is not None:
            samples["wind_direction_deg"].append(wind_direction)
        seen += 1
        if seen >= count:
            break
    return samples


def _print_field_readiness_checklist() -> None:
    """Dump ``docs/FIELD_READINESS.md`` to stdout (verbose mode).

    The checklist is kept as a separate markdown file so field crews
    can also read it offline; ``--verbose`` just concatenates it to the
    automated health-check output.
    """
    checklist = Path(__file__).resolve().parents[2] / "docs" / "FIELD_READINESS.md"
    if checklist.exists():
        print(checklist.read_text())
        print("─" * 72)
    else:
        print("(FIELD_READINESS.md not found)")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="EDGE-AURA-MFP health check")
    p.add_argument("--input-mode", choices=["serial", "file"], default="serial")
    p.add_argument("--input", default=None)
    p.add_argument("--port",  default="/dev/serial0")
    p.add_argument("--baud",  type=int, default=115_200)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--mock", action="store_true",
                   help="Run all 5 pre-flight checks against mock inputs "
                        "(no hardware required).")
    p.add_argument("--verbose", action="store_true",
                   help="Also print docs/FIELD_READINESS.md so the full "
                        "pre-flight checklist comes out with the results.")
    args = p.parse_args(argv)

    if args.verbose:
        _print_field_readiness_checklist()

    if args.mock:
        return _run_mock_checks()

    if args.input_mode == "file":
        if not args.input:
            p.error("--input required with --input-mode=file")
        reader = SerialReader.from_file(args.input)
    else:
        reader = SerialReader(port=args.port, baud=args.baud)

    with reader:
        samples = _collect(reader, Calibration(), args.count)

    print("Health check results:")
    overall = True
    for name, lo, hi in _RANGES:
        vals = samples[name]
        if not vals:
            print(f"  [--] {name}: no data")
            overall = False
            continue
        mean = statistics.fmean(vals)
        ok = lo <= mean <= hi
        overall &= ok
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag}] {name}: {mean:.3f}  (expected {lo}–{hi})")
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


# ══════════════════════════════════════════════════════════════════════
# Mock pre-flight checks (Edge-Batch B gate #4)
# ══════════════════════════════════════════════════════════════════════
#
# Runs the 5 health checks defined in the project plan against
# synthetic inputs:
#
#   1. Serial link: simulate 5 s of frames at ≥1 Hz → expect ≥ 4.
#   2. CRC: parse each frame via the real serial_reader stack.
#   3. Calibration sanity: pyranometer > 0, T_amb ∈ [-20, 60], WS ∈ [0, 50].
#   4. Camera: one SkyCameraCapture call, file exists and > 10 KB.
#   5. Pass/fail report + exit code.

def _run_mock_checks() -> int:  # noqa: C901 – linear sequence, readability wins.
    from pi.calibration import CalibrationLoader
    from pi.camera import SkyCameraCapture
    from pi.packet_builder import build_sensor_packet, validate_sensor_packet
    from scripts.generate_fixture import FIXTURE_PATH

    print("EDGE-AURA-MFP health-check (mock mode)\n")
    results: list[tuple[str, bool, str]] = []

    # 1 – Serial frames at ≥ 1 Hz
    t0 = time.time()
    frames: list[dict | None] = []
    with SerialReader.from_file(str(FIXTURE_PATH)) as reader:
        for frame in reader:
            frames.append(frame)
            # Simulate real-time by bailing after a nominal "5 s".
            if len(frames) >= 5:
                break
    good = sum(1 for f in frames if f is not None)
    results.append(("frames received ≥ 4", good >= 4, f"{good} frames"))

    # 2 – CRC: serial_reader already verified CRC for each frame.
    crc_ok = all(f is not None for f in frames)
    results.append(("CRC OK on every frame", crc_ok, "reader returned no Nones"))

    # 3 – Calibration sanity
    cal = Calibration()
    loader = CalibrationLoader()  # exercises the new loader stack
    last = next((f for f in reversed(frames) if f is not None), None)
    in_range = False
    reason = "no usable frame"
    if last is not None:
        flags = last["fault_flags"]
        g = cal.pyranometer(last["pyranometer_raw"], flags) or 0.0
        tc = cal.thermocouple(last["thermocouple_raw"], flags)
        ws = cal.anemometer(
            last["anemometer_speed_x100"], last["anemometer_dir_deg"], flags,
        )[0] or 0.0
        t_amb = next((t for t in tc if t is not None), 0.0)
        checks = [
            ("pyranometer > 0", g > 0),
            ("T_amb in [-20, 60]", -20.0 <= t_amb <= 60.0),
            ("WS in [0, 50]",     0.0 <= ws <= 50.0),
        ]
        in_range = all(ok for _, ok in checks)
        reason = "; ".join(f"{name}={ok}" for name, ok in checks)
        reason += f"  (uncalibrated: {sorted(loader.uncalibrated)})"
    results.append(("calibrated values in range", in_range, reason))

    # 4 – Camera
    with tempfile.TemporaryDirectory() as td:
        cam = SkyCameraCapture(resolution=(1280, 720), framerate=1,
                               image_dir=Path(td), force_fake=True)
        try:
            path = cam.capture_once()
        finally:
            cam.stop()
        cam_ok = (
            path is not None
            and path.exists()
            and path.stat().st_size > 10 * 1024
        )
        cam_msg = "no image" if path is None else f"{path.stat().st_size} B"
    results.append(("camera image > 10 KB", cam_ok, cam_msg))

    # 5 – Packet builder produces schema-valid output
    pkt_ok = True
    pkt_msg = "validated"
    try:
        pkt = build_sensor_packet(
            t_s=float(last["timestamp_ms"]) / 1000.0 if last else 0.0,
            G_poa=521.0, T_amb=23.5, WS=2.4,
            lat=36.53, lon=-87.36, fault_flags=0,
            sky_image_path="images/mock.jpg",
        )
        validate_sensor_packet(pkt)
    except Exception as exc:  # pylint: disable=broad-except
        pkt_ok = False
        pkt_msg = f"{exc}"
    results.append(("SensorPacket schema valid", pkt_ok, pkt_msg))

    # Report
    width = max(len(name) for name, _, _ in results)
    overall = True
    for name, ok, detail in results:
        tag = "PASS" if ok else "FAIL"
        overall &= ok
        print(f"  [{tag}] {name.ljust(width)}  {detail}")
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}  ({time.time() - t0:.1f}s)")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
