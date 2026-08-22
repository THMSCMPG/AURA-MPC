"""workstation/scripts/calibrate.py – Interactive calibration wizard (Edge-Batch B).

Subcommands
-----------

    calibrate t_amb          --reference-celsius <float> [--samples 30]
    calibrate ws              --reference-ms      <float> [--samples 30]
    calibrate thermocouple    --channel 0..4 --reference-celsius <float> [--samples 30]
    calibrate anemometer      --reference-ms   <float> [--samples 30]
    calibrate list
    calibrate verify        [--samples 10]

pyranometer subcommand REMOVED (2026-08-22) -- irradiance is manual-entry
on the workstation now, not sensed at all. t_amb/ws subcommands ADDED --
these are the two RP2040 12-bit ADC channels the current firmware
actually reads (confirmed 2026-08-22: calibration lives on the
workstation, the Pico only emits raw counts -- see calibration.py's
module docstring for the full architecture). anemometer left as-is,
NOT confirmed dead the way pyranometer was -- flagged as likely
superseded by the SparkFun Weather Meter Kit plan but not removed
without being asked; wind/rain/vane wiring is still deliberately
deferred until that hardware's in hand.

thermocouple --channel range extended 0..3 -> 0..4 -- confirmed
2026-08-22: 5 real thermocouples, not 4. Live-data collection for this
subcommand isn't wired to anything yet either way (the C firmware
doesn't read thermocouples at all until the 5th chip-select pin is
assigned -- separately deferred, see checklist).

Each wizard collects N live raw readings (or synthetic readings with
``--mock``) while the operator holds a reference instrument steady at
``reference_*``, fits a linear mapping ``physical = slope × raw + intercept``,
sanity-checks the slope against the datasheet expectation (must be within
[0.5×, 2×] or ``--force`` is required), and writes the result to
``calibration/<sensor>.json``.

File format
-----------

```
{
  "sensor":               "t_amb",
  "method":               "linear",
  "slope":                0.0192,
  "intercept":            0.0,
  "r_squared":            0.9987,
  "samples":              30,
  "reference_instrument": "Fluke 51-II S/N 12345",
  "reference_value":      22.5,
  "units":                "degC",
  "calibrated_at":        "2026-07-15T14:32:10Z",
  "calibrated_by":        "thmscmpg",
  "git_sha":              "abc1234"
}
```
"""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from workstation.calibration import (
    CalibrationLoader,
    EXPECTED_SLOPES,
    SENSOR_UNITS,
    calibration_dir,
    check_slope_sanity,
    linear_fit,
)

log = logging.getLogger("edge-aura.calibrate")

# ── Mock raw-sample generators ────────────────────────────────────────
# Given a reference value and the datasheet-expected slope, produce a
# plausible stream of raw ADC readings with ±1 % noise.

_NOISE_FRACTION = 0.01


def _mock_raws(expected_slope: float, reference: float, n: int) -> list[float]:
    if expected_slope == 0:
        raise ValueError("cannot mock: expected slope is 0")
    center = reference / expected_slope
    rng = random.Random(0xA06E)   # deterministic for CI
    return [center * (1.0 + rng.uniform(-_NOISE_FRACTION, _NOISE_FRACTION))
            for _ in range(n)]


# ── Git SHA lookup ────────────────────────────────────────────────────

def _git_sha() -> str:
    """Return the short SHA of the calibration script's git commit.

    Falls back to ``"unknown"`` when the repo is not a git checkout.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.decode().strip() or "unknown"


def _utc_iso() -> str:
    """ISO-8601 UTC timestamp with seconds precision, ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Live sample collector (calibration & verify) ──────────────────────

def _collect_live_samples(sensor_getter, n: int, *, interval: float = 0.25) -> list[float]:
    """Call ``sensor_getter()`` N times with a short pause between calls."""
    samples: list[float] = []
    for i in range(n):
        samples.append(float(sensor_getter()))
        if i + 1 < n:
            time.sleep(interval)
    return samples


# ══════════════════════════════════════════════════════════════════════
# Shared save/sanity-check pipeline
# ══════════════════════════════════════════════════════════════════════

def _maybe_save(
    *,
    sensor: str,
    samples: Iterable[float],
    reference_value: float,
    reference_instrument: str,
    units: str,
    fit_intercept: bool,
    force: bool,
    out_dir: Path,
) -> tuple[dict, Path, Optional[str]]:
    """Fit, sanity-check, and (unless vetoed) save the calibration file.

    Returns ``(record, path, warning_msg)``. ``warning_msg`` is non-None
    when the slope failed the 0.5×–2× sanity check; if ``--force`` was
    supplied the file is still written.
    """
    raws = list(samples)
    fit = linear_fit(raws, reference_value, intercept=fit_intercept)
    warning = check_slope_sanity(sensor, fit["slope"])

    record = {
        "sensor":               sensor,
        "method":               "linear",
        "slope":                fit["slope"],
        "intercept":            fit["intercept"],
        "r_squared":            fit["r_squared"],
        "samples":              fit["samples"],
        "reference_instrument": reference_instrument,
        "reference_value":      float(reference_value),
        "units":                units,
        "calibrated_at":        _utc_iso(),
        "calibrated_by":        os.environ.get("USER") or getpass.getuser(),
        "git_sha":              _git_sha(),
    }

    out_path = out_dir / f"{sensor}.json"
    if warning is not None and not force:
        # Do NOT save; let the caller emit the warning and exit non-zero.
        return record, out_path, warning

    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return record, out_path, warning


# ══════════════════════════════════════════════════════════════════════
# Subcommand handlers
# ══════════════════════════════════════════════════════════════════════

def _cmd_t_amb(args) -> int:
    sensor = "t_amb"
    if args.mock:
        raws = _mock_raws(EXPECTED_SLOPES[sensor], args.reference_celsius, args.samples)
    else:
        raws = _collect_live_raw(sensor, args.samples, args)
    return _finish(sensor, raws, args.reference_celsius,
                   reference_instrument=args.reference_instrument
                   or "unspecified thermometer",
                   units=SENSOR_UNITS[sensor],
                   fit_intercept=args.intercept, force=args.force,
                   out_dir=Path(args.cal_dir))


def _cmd_ws(args) -> int:
    sensor = "ws"
    if args.mock:
        raws = _mock_raws(EXPECTED_SLOPES[sensor], args.reference_ms, args.samples)
    else:
        raws = _collect_live_raw(sensor, args.samples, args)
    return _finish(sensor, raws, args.reference_ms,
                   reference_instrument=args.reference_instrument
                   or "unspecified anemometer reference",
                   units=SENSOR_UNITS[sensor],
                   fit_intercept=args.intercept, force=args.force,
                   out_dir=Path(args.cal_dir))


def _cmd_thermocouple(args) -> int:
    sensor = f"thermocouple_{args.channel}"
    if args.mock:
        raws = _mock_raws(EXPECTED_SLOPES[sensor], args.reference_celsius, args.samples)
    else:
        raws = _collect_live_raw(sensor, args.samples, args, channel=args.channel)
    return _finish(sensor, raws, args.reference_celsius,
                   reference_instrument=args.reference_instrument
                   or "unspecified thermometer",
                   units=SENSOR_UNITS[sensor],
                   fit_intercept=args.intercept, force=args.force,
                   out_dir=Path(args.cal_dir))


def _cmd_anemometer(args) -> int:
    sensor = "anemometer"
    if args.mock:
        raws = _mock_raws(EXPECTED_SLOPES[sensor], args.reference_ms, args.samples)
    else:
        raws = _collect_live_raw(sensor, args.samples, args)
    return _finish(sensor, raws, args.reference_ms,
                   reference_instrument=args.reference_instrument
                   or "unspecified anemometer",
                   units=SENSOR_UNITS[sensor],
                   fit_intercept=args.intercept, force=args.force,
                   out_dir=Path(args.cal_dir))


def _cmd_list(args) -> int:
    loader = CalibrationLoader(cal_dir=Path(args.cal_dir))
    rows = loader.summary()
    width = max(len(r["sensor"]) for r in rows)
    print(f"Calibration files in {args.cal_dir}:\n")
    print(f"  {'sensor'.ljust(width)}  status  slope         timestamp")
    print(f"  {'-' * width}  ------  ------------  --------------------")
    for r in rows:
        status = "OK" if r["calibrated"] else "MISSING"
        slope = "          --" if r["slope"] is None else f"{r['slope']:.6g}".rjust(12)
        ts = r["calibrated_at"] or "--"
        print(f"  {r['sensor'].ljust(width)}  {status:6s}  {slope}  {ts}")
    missing = [r["sensor"] for r in rows if not r["calibrated"]]
    if missing:
        print(f"\nMissing: {', '.join(missing)} – run the wizard to calibrate.")
        return 1
    return 0


def _cmd_verify(args) -> int:
    """Run a few live reads through the current calibration and ask for confirmation."""
    loader = CalibrationLoader(cal_dir=Path(args.cal_dir))
    print("Verification – 10 live readings through the current calibration:")
    print(f"(calibrated sensors: "
          f"{len([s for s in loader.calibrations if loader.is_calibrated(s)])}"
          f"  uncalibrated: {sorted(loader.uncalibrated)})")

    if args.mock:
        # Synthesise plausible readings from the expected slopes.
        rng = random.Random(0xA06E)
        for i in range(args.samples):
            print(f"  [{i + 1:2d}] "
                  f"T_amb = {loader.apply('t_amb', rng.uniform(0, 4095)):5.1f} degC   "
                  f"WS    = {loader.apply('ws', rng.uniform(0, 4095)):4.2f} m/s   "
                  f"TC0   = {loader.apply('thermocouple_0', rng.uniform(2000, 4000)):5.1f} degC")
        print("\nMock verify always PASSes.")
        return 0

    # Live path: STALE, not rewritten to the new raw-packet JSON-line
    # format yet -- see calibrate.py's module docstring. Raises
    # ImportError here (no workstation.serial_reader anymore) rather than
    # continuing with the old, no-longer-matching raw-frame field names.
    from workstation.serial_reader import SerialReader  # noqa: PLC0415
    reader = SerialReader(port=args.port, baud=args.baud)
    count = 0
    with reader:
        for frame in reader:
            if frame is None:
                continue
            count += 1
            t = loader.apply("t_amb", frame["T_amb_raw"])
            w = loader.apply("ws", frame["WS_raw"])
            print(f"  [{count:2d}]  T_amb={t:5.1f}  WS={w:4.2f}")
            if count >= args.samples:
                break
    resp = input("\nDo these values look sane? [y/N] ").strip().lower()
    return 0 if resp == "y" else 1


# ── Live raw-sample path (STALE -- raises ImportError, see module docstring) ──

def _collect_live_raw(sensor: str, n: int, args, *, channel: int = 0) -> list[float]:  # pragma: no cover
    from workstation.serial_reader import SerialReader  # noqa: PLC0415 -- see module docstring, no longer exists
    print(f"Collecting {n} raw readings for {sensor}. "
          "Hold the reference instrument steady …")
    reader = SerialReader(port=args.port, baud=args.baud)
    samples: list[float] = []
    with reader:
        for frame in reader:
            if frame is None:
                continue
            if sensor == "t_amb":
                samples.append(float(frame["T_amb_raw"]))
            elif sensor == "ws":
                samples.append(float(frame["WS_raw"]))
            elif sensor.startswith("thermocouple_"):
                samples.append(float(frame["thermocouple_raw"][channel]))
            elif sensor == "anemometer":
                samples.append(float(frame["anemometer_speed_x100"]))
            print(f"  [{len(samples):2d}] raw={samples[-1]:10.1f}")
            if len(samples) >= n:
                break
    return samples


# ── Final save/report ─────────────────────────────────────────────────

def _finish(sensor, raws, reference_value, *, reference_instrument, units,
            fit_intercept, force, out_dir) -> int:
    record, out_path, warning = _maybe_save(
        sensor=sensor, samples=raws, reference_value=reference_value,
        reference_instrument=reference_instrument, units=units,
        fit_intercept=fit_intercept, force=force, out_dir=out_dir,
    )
    print(f"\nFit for {sensor}: slope={record['slope']:.6g}  "
          f"intercept={record['intercept']:.6g}  "
          f"r²={record['r_squared']:.4f}  samples={record['samples']}")
    if warning is not None:
        print(f"WARNING: {warning}", file=sys.stderr)
        if not force:
            print("Refusing to save without --force.", file=sys.stderr)
            return 2
        print("(Saved anyway because --force was supplied.)", file=sys.stderr)
    print(f"Saved calibration → {out_path}")
    return 0


# ══════════════════════════════════════════════════════════════════════
# CLI wiring
# ══════════════════════════════════════════════════════════════════════

def _add_common(sub):
    sub.add_argument("--samples", type=int, default=30,
                     help="Number of raw samples to average (default: 30).")
    sub.add_argument("--port", default="/dev/serial0")
    sub.add_argument("--baud", type=int, default=115_200)
    sub.add_argument("--mock", action="store_true",
                     help="Synthesise raw samples instead of reading the serial link.")
    sub.add_argument("--force", action="store_true",
                     help="Save even if the fitted slope fails the 0.5×–2× sanity check.")
    sub.add_argument("--intercept", action="store_true",
                     help="Fit a two-parameter linear model (default: zero-intercept).")
    sub.add_argument("--reference-instrument", default=None,
                     help="Free-form reference instrument description (make/model/SN).")
    sub.add_argument("--cal-dir", default=str(calibration_dir()),
                     help="Output directory for calibration JSON files.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calibrate",
                                description="EDGE-AURA-MFP calibration wizard")
    subs = p.add_subparsers(dest="cmd", required=True)

    ta = subs.add_parser("t_amb", help="Calibrate the ambient temperature ADC channel (degC)")
    ta.add_argument("--reference-celsius", type=float, required=True)
    _add_common(ta)
    ta.set_defaults(func=_cmd_t_amb)

    tw = subs.add_parser("ws", help="Calibrate the wind speed ADC channel (m/s)")
    tw.add_argument("--reference-ms", type=float, required=True)
    _add_common(tw)
    tw.set_defaults(func=_cmd_ws)

    st = subs.add_parser("thermocouple", help="Calibrate one thermocouple channel (°C)")
    st.add_argument("--channel", type=int, choices=[0, 1, 2, 3, 4], required=True)
    st.add_argument("--reference-celsius", type=float, required=True)
    _add_common(st)
    st.set_defaults(func=_cmd_thermocouple)

    sa = subs.add_parser("anemometer", help="Calibrate the anemometer (m/s) -- likely superseded, see module docstring")
    sa.add_argument("--reference-ms", type=float, required=True)
    _add_common(sa)
    sa.set_defaults(func=_cmd_anemometer)

    sl = subs.add_parser("list", help="List current calibration files")
    sl.add_argument("--cal-dir", default=str(calibration_dir()))
    sl.set_defaults(func=_cmd_list)

    sv = subs.add_parser("verify",
                         help="Take N live readings through current calibration")
    sv.add_argument("--samples", type=int, default=10)
    sv.add_argument("--port", default="/dev/serial0")
    sv.add_argument("--baud", type=int, default=115_200)
    sv.add_argument("--mock", action="store_true")
    sv.add_argument("--cal-dir", default=str(calibration_dir()))
    sv.set_defaults(func=_cmd_verify)

    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
