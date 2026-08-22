#!/bin/bash
#SBATCH --job-name=rk4tran_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=08:00:00
##SBATCH --partition=PARTITION_NAME_HERE   # <-- fill in, see generate_chunks.slurm

# The ONE persistent trainer process. Submit this alongside (not instead
# of) generate_chunks.slurm's array job -- they both point at the SAME
# WATCH_DIR on Loki's shared NFS filesystem, so this watcher picks up
# chunks as the array tasks produce them, with no explicit file transfer
# needed between nodes.
#
# --time=08:00:00 matches your 8hr/week self-imposed budget for this
# project (out of a 20hr/week personal cap, leaving 12hr for coursework).
#
# REAL THROUGHPUT NUMBERS (measured, not estimated -- see checklist Section
# 2 for the full benchmark story, including a real bug found and fixed
# along the way): at epochs-per-chunk=1, training runs at ~0.95x
# generation's own pace (17.2 min/location vs 18.2 min/location, measured
# on 1.76M real rows and extrapolated to a full ~38M-row location). This
# means a SINGLE generator can very nearly keep this single trainer fully
# fed on its own -- and since only ONE trainer can safely exist (the
# checkpoint-write-race problem), MORE than a few generator nodes doesn't
# increase total pipeline throughput, it just builds up an unconsumed
# backlog. See generate_chunks.slurm for the recommended array concurrency
# given this.
#
# --idle-timeout below (30 min) means this exits cleanly once no new
# chunks have appeared for a while, rather than sitting idle burning your
# walltime allocation if generation finishes early.
#
# MUST run generate_holdout_val.slurm first and wait for it to complete --
# this job will exit immediately with an error if the held-out validation
# memmap doesn't exist yet (see streaming_trainer_watcher.py).

set -euo pipefail

export RUN_DIR="${RUN_DIR:-$HOME/aura_mpc_run}"

mkdir -p logs

python3 tools/streaming_trainer_watcher.py \
    --watch-dir "$RUN_DIR/lattice_batches" \
    --val-memmap-meta "$RUN_DIR/memmaps/holdout_val.meta.json" \
    --checkpoint-dir "$RUN_DIR/checkpoints" \
    --raw-agg-dir "$RUN_DIR/raw_aggregates" \
    --epochs-per-chunk 1 \
    --poll-interval 30 \
    --idle-timeout 1800
