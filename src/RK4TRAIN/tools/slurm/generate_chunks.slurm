#!/bin/bash
#SBATCH --job-name=rk4tran_gen
#SBATCH --output=logs/gen_%A_%a.out
#SBATCH --error=logs/gen_%A_%a.err
#SBATCH --array=1-90%3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
##SBATCH --partition=PARTITION_NAME_HERE   # <-- fill in: run `sinfo` on Loki to see partition names, uncomment

# Generates location chunks in parallel across the cluster, drawing from a
# RANDOMIZED per-sprint subset of the ~1050-point location pool (not a
# fixed contiguous range) -- see tools/select_training_locations.py, which
# you must run ONCE before submitting this array job.
#
# Each array task handles LOCS_PER_TASK locations (default 1) pulled from
# that pre-selected, randomized list -- 90 array tasks for a 90-location
# sprint by default, matching select_training_locations.py's own default.
# Reserved validation indices (the pool's first 12 points) are never
# eligible for selection -- enforced in select_training_locations.py, not
# just documented.
#
# --array=1-90%3 : 90 total tasks, at most 3 running concurrently. This is
#   NOT a hardware limit (Loki has 32 nodes) -- it's a throughput-matching
#   choice, and a real one, not a guess. Measured (see checklist Section
#   1.11 for the full story, including a real data-loading bug found and
#   fixed along the way): with the fix in place, ONE trainer running at
#   epochs-per-chunk=1 processes chunks at ~0.95x generation's own pace
#   (17.2 min/location vs 18.2 min/location). Since only ONE trainer can
#   safely exist at a time (concurrent writers would corrupt the shared
#   model checkpoint -- see streaming_trainer_watcher.py), a single
#   generator can very nearly keep it fed on its own. 3 generators (not 1)
#   gives headroom against real per-location variance (high-latitude
#   summer days generate far more hour-samples than short winter days) so
#   the trainer's queue doesn't run dry between chunks, without wasting
#   node-hours on backlog the trainer can't consume anyway -- also happens
#   to match your own 4-node student-partition cap (3 generators + 1
#   trainer = 4).
#
# Real per-location generation time (v3, full resolution, single core):
#   ~18.2 min/location average -- with --time=04:00:00 per task there's
#   ample headroom even for the longest individual locations.
#
# BEFORE running this:
#   1. Build RK4TRAN: cd src/RK4TRAIN && bash make.sh
#   2. Run generate_holdout_val.slurm first, wait for it to complete.
#   3. Run tools/select_training_locations.py (plain python3, no SLURM
#      needed -- it's instant, no Fortran generation involved) to produce
#      this sprint's randomized location file:
#        python3 tools/select_training_locations.py \
#            --pool-meta lattice_pools_generated.f90 \
#            --n-locations 90 --out $RUN_DIR/sprint_locations.txt \
#            --log-file $RUN_DIR/location_selection_history.jsonl
#   4. Fill in the partition name above (run `sinfo` on Loki).

export RK4TRAIN_DIR="${RK4TRAIN_DIR:-$HOME/AURA-MPC/src/RK4TRAIN}"
export WATCH_DIR="${WATCH_DIR:-$HOME/aura_mpc_run/lattice_batches}"
export SPRINT_LOCATIONS_FILE="${SPRINT_LOCATIONS_FILE:-$HOME/aura_mpc_run/sprint_locations.txt}"
export LOCS_PER_TASK="${LOCS_PER_TASK:-1}"

mkdir -p logs
bash tools/slurm/generate_chunk_task.sh
