#!/usr/bin/env python3
"""merge_chunk_aggregates.py

Combines the raw per-bin accumulators (sum, sumsq, count) written by
preprocess_for_plots.awk across MULTIPLE chunks into final mean/stddev
.dat files -- the same format plot_synthetic_data.gp expects, so gnuplot
doesn't need to know or care whether the data came from one file or a
thousand streamed-and-deleted chunks.

Combining raw sums is mathematically EXACT (identical to running the
aggregation once on the full concatenated dataset) -- this is why
preprocess_for_plots.awk emits raw accumulators (.raw) instead of
pre-computed means: averaging already-computed means across chunks would
NOT be correct in general (needs count-weighting at minimum, and stddev
can't be recovered from a stddev-of-stddevs at all).

Designed to be run repeatedly as new chunks complete (incremental,
idempotent) OR once at the end after all chunks are processed -- either
way, point it at every .raw file produced so far and it recomputes the
final .dat outputs from scratch each time. Cheap: total size of all raw
accumulator files stays tiny (bounded by bin count) even after processing
a multi-TB dataset's worth of chunks.

Usage:
    python3 merge_chunk_aggregates.py --raw-dir /path/to/chunk/aggregates/ --out-dir .

Expects files named like "chunk_003_plot1_diurnal.raw" (any prefix before
"plotN_...raw" is fine -- glob matches on the "plotN_...raw" suffix).
"""

from __future__ import annotations

import argparse
import glob
import math
from collections import defaultdict
from pathlib import Path


def merge_plot1(raw_dir: Path, out_dir: Path) -> None:
    """hour_bin sum_T sumsq_T sum_eta sumsq_eta n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot1_diurnal.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, sT, ssT, seta, sseta, n = line.split()
                a = acc[int(b)]
                a[0] += float(sT); a[1] += float(ssT); a[2] += float(seta); a[3] += float(sseta); a[4] += int(n)
    with open(out_dir / "plot1_diurnal.dat", "w") as out:
        out.write("# hour mean_T_operating_C stddev_T mean_eta stddev_eta n\n")
        for b in sorted(acc):
            sT, ssT, seta, sseta, n = acc[b]
            if n == 0:
                continue
            mT = sT / n
            sdT = math.sqrt(max(0.0, ssT / n - mT * mT))
            meta = seta / n
            sdeta = math.sqrt(max(0.0, sseta / n - meta * meta))
            out.write(f"{b/2.0:.2f} {mT:.4f} {sdT:.4f} {meta:.6f} {sdeta:.6f} {n}\n")


def merge_plot2(raw_dir: Path, out_dir: Path) -> None:
    """tamb_bin sum_eta n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot2_eta_vs_tamb.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, seta, n = line.split()
                a = acc[int(b)]
                a[0] += float(seta); a[1] += int(n)
    with open(out_dir / "plot2_eta_vs_tamb.dat", "w") as out:
        out.write("# T_amb_C mean_eta n\n")
        for b in sorted(acc):
            seta, n = acc[b]
            if n == 0:
                continue
            out.write(f"{-50.0 + b*2.0 + 1.0:.2f} {seta/n:.6f} {n}\n")


def merge_plot3(raw_dir: Path, out_dir: Path) -> None:
    """latitude(string key) sum_pitch n"""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot3_optimal_pitch_vs_lat.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                lat, spitch, n = line.split()
                a = acc[lat]
                a[0] += float(spitch); a[1] += int(n)
    with open(out_dir / "plot3_optimal_pitch_vs_lat.dat", "w") as out:
        out.write("# latitude mean_optimal_pitch n\n")
        for lat in sorted(acc, key=float):
            spitch, n = acc[lat]
            if n == 0:
                continue
            out.write(f"{lat} {spitch/n:.4f} {n}\n")


def merge_plot4(raw_dir: Path, out_dir: Path) -> None:
    """orient_bin sum_eta sumsq_eta n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot4_eta_vs_orientation_error.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, seta, sseta, n = line.split()
                a = acc[int(b)]
                a[0] += float(seta); a[1] += float(sseta); a[2] += int(n)
    with open(out_dir / "plot4_eta_vs_orientation_error.dat", "w") as out:
        out.write("# orientation_error_deg mean_eta stddev_eta n\n")
        for b in sorted(acc):
            seta, sseta, n = acc[b]
            if n == 0:
                continue
            me = seta / n
            sde = math.sqrt(max(0.0, sseta / n - me * me))
            out.write(f"{b + 0.5:.1f} {me:.6f} {sde:.6f} {n}\n")


def merge_plot5(raw_dir: Path, out_dir: Path) -> None:
    """cloud_bin sum_eta sum_sigma n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot5_eta_vs_cloud.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, seta, ssig, n = line.split()
                a = acc[int(b)]
                a[0] += float(seta); a[1] += float(ssig); a[2] += int(n)
    with open(out_dir / "plot5_eta_vs_cloud.dat", "w") as out:
        out.write("# cloud_cover mean_eta mean_T_operating_sigma n\n")
        for b in sorted(acc):
            seta, ssig, n = acc[b]
            if n == 0:
                continue
            out.write(f"{b*0.05 + 0.025:.3f} {seta/n:.6f} {ssig/n:.4f} {n}\n")


def merge_plot6(raw_dir: Path, out_dir: Path) -> None:
    """relax_bin sum_delta n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot6_thermal_relaxation.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, sdelta, n = line.split()
                a = acc[int(b)]
                a[0] += float(sdelta); a[1] += int(n)
    with open(out_dir / "plot6_thermal_relaxation.dat", "w") as out:
        out.write("# delta_from_ss_C mean_15min_change_C n\n")
        for b in sorted(acc):
            sdelta, n = acc[b]
            if n == 0:
                continue
            out.write(f"{-150.0 + b*5.0 + 2.5:.2f} {sdelta/n:.4f} {n}\n")


def merge_plot7(raw_dir: Path, out_dir: Path) -> None:
    """alt_bin sum_T sum_eta n"""
    acc: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot7_vs_elevation.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                b, sT, seta, n = line.split()
                a = acc[int(b)]
                a[0] += float(sT); a[1] += float(seta); a[2] += int(n)
    with open(out_dir / "plot7_vs_elevation.dat", "w") as out:
        out.write("# elevation_m mean_T_operating_C mean_eta n\n")
        for b in sorted(acc):
            sT, seta, n = acc[b]
            if n == 0:
                continue
            out.write(f"{b*200.0 + 100.0:.1f} {sT/n:.4f} {seta/n:.6f} {n}\n")


def merge_plot8(raw_dir: Path, out_dir: Path) -> None:
    """wind_speed(string key) sum_sigma n"""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for f in glob.glob(str(raw_dir / "*plot8_sigma_vs_wind.raw")):
        with open(f) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                wind, ssig, n = line.split()
                a = acc[wind]
                a[0] += float(ssig); a[1] += int(n)
    with open(out_dir / "plot8_sigma_vs_wind.dat", "w") as out:
        out.write("# wind_speed mean_T_operating_sigma n\n")
        for wind in sorted(acc, key=float):
            ssig, n = acc[wind]
            if n == 0:
                continue
            out.write(f"{wind} {ssig/n:.4f} {n}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, required=True, help="Directory containing all *.raw chunk accumulator files")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="Where to write the final plotN_*.dat files")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_raw = len(list(args.raw_dir.glob("*.raw")))
    print(f"Merging {n_raw} raw accumulator files from {args.raw_dir}...")

    merge_plot1(args.raw_dir, args.out_dir)
    merge_plot2(args.raw_dir, args.out_dir)
    merge_plot3(args.raw_dir, args.out_dir)
    merge_plot4(args.raw_dir, args.out_dir)
    merge_plot5(args.raw_dir, args.out_dir)
    merge_plot6(args.raw_dir, args.out_dir)
    merge_plot7(args.raw_dir, args.out_dir)
    merge_plot8(args.raw_dir, args.out_dir)

    print(f"Wrote 8 merged .dat files to {args.out_dir} -- run plot_synthetic_data.gp there to render.")


if __name__ == "__main__":
    main()
