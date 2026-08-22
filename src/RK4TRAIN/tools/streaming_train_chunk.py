#!/usr/bin/env python3
"""streaming_train_chunk.py -- train on ONE chunk, called repeatedly by the
streaming pipeline orchestrator (tools/streaming_pipeline.py).

This is deliberately NOT a full training run in itself -- it's one step of
a continual/streaming training regime: load the persistent checkpoint from
the previous chunk (or initialize fresh on the very first call), train a
few epochs on JUST this chunk's data, evaluate against a fixed held-out
validation set, save the updated checkpoint, append to a running metrics
log, and exit. The orchestrator deletes the chunk's raw data after this
returns successfully.

IMPORTANT ML NOTE: this is single/few-pass streaming training, not the
usual multi-epoch training over a fixed, fully-retained dataset. Each row
is seen only a handful of times (across however many chunks touch similar
conditions) rather than repeatedly across many epochs. This is a real,
standard trade-off for datasets too large to retain -- expect somewhat
noisier convergence than a traditional multi-epoch regime, and expect
the model to depend somewhat on chunk *order* (later chunks have more
influence, similar to a decaying learning-rate-free online learning
setup). If training instability shows up in the metrics log, the two
easiest levers are more epochs-per-chunk (--epochs) or a smaller learning
rate.

The normalizer is fit ONCE (from the held-out validation set, which is
generated first and never deleted) and reused for every chunk -- refitting
per chunk would make normalization inconsistent across the streaming run.

Usage (called by the orchestrator, or manually per chunk):
  python3 streaming_train_chunk.py \\
    --memmap-meta /path/to/chunk_003.meta.json \\
    --val-memmap-meta /path/to/holdout_val.meta.json \\
    --checkpoint-dir /path/to/persistent/checkpoints/ \\
    --normalizer-path /path/to/persistent/normalizer.json \\
    --epochs 3 --batch-size 64 --lr 1e-3
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

THIS_FILE = Path(__file__).resolve()
# THIS_FILE = .../src/RK4TRAIN/tools/streaming_train_chunk.py
ML_DIR = THIS_FILE.parents[1] / "ml"   # .../src/RK4TRAIN/ml
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

from data import NumericNormalizer
from data.memmap_dataset import RK4TRANMemmapDataset
from models import PINNSurrogate
from training import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--memmap-meta", type=Path, required=True, help="This chunk's .meta.json (from build_memmap)")
    p.add_argument("--val-memmap-meta", type=Path, required=True, help="Held-out validation .meta.json (never deleted, fixed for the whole run)")
    p.add_argument("--checkpoint-dir", type=Path, required=True, help="Where the persistent model checkpoint lives across chunks")
    p.add_argument("--normalizer-path", type=Path, required=True, help="Persistent normalizer.json -- fit once, reused every chunk")
    p.add_argument("--metrics-log", type=Path, default=None, help="CSV to append per-chunk metrics to (default: checkpoint-dir/streaming_metrics.csv)")
    p.add_argument("--epochs", type=int, default=3, help="Epochs to train on THIS chunk only")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-residual-blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--chunk-tag", type=str, default=None, help="Label for this chunk in the metrics log (default: derived from memmap-meta filename)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = args.metrics_log or (args.checkpoint_dir / "streaming_metrics.csv")
    chunk_tag = args.chunk_tag or args.memmap_meta.stem
    device = args.device

    t_start = time.time()

    # --- normalizer: fit once (from the held-out val set) and persist, reuse forever after ---
    if args.normalizer_path.exists():
        normalizer = NumericNormalizer.load(args.normalizer_path)
        print(f"Loaded existing normalizer from {args.normalizer_path}")
    else:
        print("No normalizer found -- fitting fresh from the held-out validation set (this only happens once).")
        val_ds_for_fit = RK4TRANMemmapDataset(args.val_memmap_meta)
        normalizer = NumericNormalizer()
        normalizer.fit(val_ds_for_fit.get_normalizer_data(sample_rows=None))  # val set is small/fixed, full scan is fine
        normalizer.save(args.normalizer_path)
        print(f"Fitted and saved normalizer to {args.normalizer_path}")

    input_dim = RK4TRANMemmapDataset  # placeholder, replaced below once we load the chunk

    # --- load this chunk ---
    chunk_ds = RK4TRANMemmapDataset(args.memmap_meta, normalizer=normalizer)
    input_dim = chunk_ds.get_input_dim()
    print(f"Chunk '{chunk_tag}': {len(chunk_ds):,} rows, input_dim={input_dim}")

    # --- load or initialize the persistent model ---
    checkpoint_path = args.checkpoint_dir / "streaming_model.pt"
    model = PINNSurrogate(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_residual_blocks=args.num_residual_blocks,
        num_outputs=4,
        dropout=args.dropout,
    )
    is_first_chunk = not checkpoint_path.exists()
    if not is_first_chunk:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded persistent checkpoint from {checkpoint_path}")
    else:
        print("No existing checkpoint -- initializing a fresh model (this is the first chunk).")

    trainer = Trainer(model=model, device=device, lr=args.lr, weight_decay=args.weight_decay)

    # --- train on this chunk only ---
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(chunk_ds.batch_iterator(args.batch_size, shuffle=True, seed=epoch))
        print(f"  epoch {epoch+1}/{args.epochs}: train_loss={metrics['train_loss']:.6f}")

    # --- evaluate against the FIXED held-out validation set ---
    val_ds = RK4TRANMemmapDataset(args.val_memmap_meta, normalizer=normalizer)
    val_metrics = trainer.validate(val_ds.batch_iterator(args.batch_size, shuffle=False))
    print(f"  validation (held-out, {len(val_ds):,} rows): val_loss={val_metrics['val_loss']:.6f}")

    # --- save the updated persistent checkpoint ---
    torch.save(model.state_dict(), checkpoint_path)

    # --- append to the running metrics log ---
    elapsed_s = time.time() - t_start
    log_exists = metrics_log.exists()
    with open(metrics_log, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["chunk_tag", "chunk_rows", "epochs", "val_loss", "elapsed_s"])
        writer.writerow([chunk_tag, len(chunk_ds), args.epochs, val_metrics["val_loss"], f"{elapsed_s:.1f}"])

    print(f"Chunk '{chunk_tag}' done in {elapsed_s:.1f}s. Checkpoint updated: {checkpoint_path}")
    print("Safe for the orchestrator to delete this chunk's raw data now.")


if __name__ == "__main__":
    main()
