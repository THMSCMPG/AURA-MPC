#!/usr/bin/env python3
"""streaming_trainer_watcher.py -- the single persistent consumer in the
producer/consumer split needed for real cluster parallelism.

WHY THIS EXISTS (see streaming_pipeline.py's docstring for the full
concurrency warning): multiple SLURM array tasks can safely generate
location-chunks in parallel (main.f90 --loc-range is fully independent per
chunk), but only ONE process can safely own the persistent model
checkpoint -- concurrent writers would race and corrupt it. This script IS
that one process: it polls a shared directory (works directly on Loki's
shared NFS filesystem -- no explicit cross-node file transfer needed) for
".ready" marker files that generation tasks drop when a chunk's CSV is
complete, claims and processes them one at a time (convert -> train ->
aggregate -> delete), and keeps running until told to stop or no new
chunks appear for a while.

Run this as ONE long-lived SLURM job, submitted alongside (not instead of)
a SLURM ARRAY job that runs main.f90 --loc-range across many nodes writing
into the SAME watched directory. See tools/slurm/*.slurm for both job
scripts.

Coordination protocol (simple, NFS-safe -- avoids relying on NFS's
sometimes-shaky distributed locking):
  - Generation tasks write "<chunk_tag>.csv" then, only once the CSV write
    is complete, write "<chunk_tag>.ready" (empty file) as a completion
    marker. Never process a .csv without its matching .ready file --
    this avoids the watcher racing ahead of a still-being-written file.
  - This watcher, on claiming a chunk, immediately renames the .ready
    marker to .claimed (an atomic operation on any POSIX-ish filesystem,
    including NFS for same-directory renames) BEFORE starting work, so a
    restarted watcher won't reprocess a chunk that crashed mid-processing
    -- the .claimed marker with no corresponding output means "orphaned,
    needs manual inspection," not "safe to silently redo."

Usage:
  python3 streaming_trainer_watcher.py \\
    --watch-dir /shared/lattice_batches/ \\
    --val-memmap-meta /shared/memmaps/holdout_val.meta.json \\
    --checkpoint-dir /shared/checkpoints/ \\
    --raw-agg-dir /shared/raw_aggregates/ \\
    --epochs-per-chunk 3 \\
    --poll-interval 30 \\
    --idle-timeout 1800
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
# THIS_FILE = .../src/RK4TRAIN/tools/streaming_trainer_watcher.py
REPO_ROOT = THIS_FILE.parents[3]   # tools -> RK4TRAIN -> src -> repo root


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def convert_to_memmap(csv_path: Path, memmap_dir: Path, name: str) -> Path:
    ml_dir = str(REPO_ROOT / "src" / "RK4TRAIN" / "ml")
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    from data.memmap_dataset import build_memmap
    return build_memmap([csv_path], memmap_dir, name=name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--watch-dir", type=Path, required=True, help="Shared directory that generation array tasks write chunk CSVs + .ready markers into")
    p.add_argument("--val-memmap-meta", type=Path, required=True, help="Held-out validation .meta.json -- generate this FIRST, separately, before starting the watcher")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--raw-agg-dir", type=Path, required=True)
    p.add_argument("--memmap-dir", type=Path, default=None, help="Scratch dir for per-chunk memmaps (default: watch-dir/memmaps)")
    p.add_argument("--epochs-per-chunk", type=int, default=3)
    p.add_argument("--poll-interval", type=int, default=30, help="Seconds between checks for new .ready chunks")
    p.add_argument("--idle-timeout", type=int, default=1800, help="Exit if no new chunks appear for this many seconds (0 = never exit, poll forever)")
    p.add_argument("--keep-raw-csv", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    memmap_dir = args.memmap_dir or (args.watch_dir / "memmaps")
    memmap_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.raw_agg_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = args.checkpoint_dir / "normalizer.json"

    if not args.val_memmap_meta.exists():
        sys.exit(
            f"Held-out validation memmap not found: {args.val_memmap_meta}\n"
            "Generate and convert it FIRST (before starting this watcher) -- "
            "see tools/slurm/generate_holdout_val.slurm."
        )

    print(f"Watching {args.watch_dir} for .ready chunk markers (poll every {args.poll_interval}s)...")
    n_processed = 0
    last_activity = time.time()

    while True:
        ready_files = sorted(args.watch_dir.glob("*.ready"))
        if not ready_files:
            idle_for = time.time() - last_activity
            if args.idle_timeout and idle_for > args.idle_timeout:
                print(f"No new chunks for {idle_for:.0f}s (idle-timeout={args.idle_timeout}s) -- exiting.")
                break
            time.sleep(args.poll_interval)
            continue

        ready_marker = ready_files[0]
        chunk_tag = ready_marker.stem
        csv_path = args.watch_dir / f"{chunk_tag}.csv"
        claimed_marker = args.watch_dir / f"{chunk_tag}.claimed"

        if not csv_path.exists():
            print(f"WARNING: {ready_marker} has no matching CSV ({csv_path}) -- skipping, leaving marker for manual inspection.")
            ready_marker.rename(args.watch_dir / f"{chunk_tag}.orphaned")
            continue

        # Atomic claim: rename .ready -> .claimed BEFORE starting work, so a
        # crash mid-processing leaves an unambiguous "orphaned, needs
        # inspection" marker instead of silently being reprocessed or lost.
        ready_marker.rename(claimed_marker)
        last_activity = time.time()

        print(f"\n=== Processing {chunk_tag} ===")
        try:
            t0 = time.time()
            chunk_meta = convert_to_memmap(csv_path, memmap_dir, chunk_tag)
            print(f"  converted to memmap in {time.time()-t0:.1f}s")

            t0 = time.time()
            run([
                sys.executable, str(THIS_FILE.parent / "streaming_train_chunk.py"),
                "--memmap-meta", str(chunk_meta),
                "--val-memmap-meta", str(args.val_memmap_meta),
                "--checkpoint-dir", str(args.checkpoint_dir),
                "--normalizer-path", str(normalizer_path),
                "--epochs", str(args.epochs_per_chunk),
                "--chunk-tag", chunk_tag,
            ])
            print(f"  trained in {time.time()-t0:.1f}s")

            t0 = time.time()
            run(["awk", "-f", str(THIS_FILE.parent / "preprocess_for_plots.awk"),
                 "-v", f"out_prefix={args.raw_agg_dir}/{chunk_tag}_", str(csv_path)])
            print(f"  aggregated plot stats in {time.time()-t0:.1f}s")

            if not args.keep_raw_csv:
                csv_path.unlink()
                (memmap_dir / f"{chunk_tag}.dat").unlink(missing_ok=True)
                chunk_meta.unlink(missing_ok=True)
                print(f"  deleted raw chunk data")

            claimed_marker.unlink()  # success -- clean up the marker entirely
            n_processed += 1
            print(f"  done. ({n_processed} chunks processed so far)")

        except Exception as e:
            print(f"ERROR processing {chunk_tag}: {e}")
            print(f"  leaving {claimed_marker} in place for manual inspection -- NOT auto-retrying.")
            # deliberately don't delete the .claimed marker or re-raise --
            # a bad chunk shouldn't take down the whole watcher, but it
            # also shouldn't be silently retried/lost.

    print(f"\nWatcher exiting. Total chunks processed: {n_processed}")


if __name__ == "__main__":
    main()
