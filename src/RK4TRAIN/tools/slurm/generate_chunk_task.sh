#!/usr/bin/env bash
# generate_chunk_task.sh -- run by ONE SLURM array task. Extracts this
# task's slice of location indices from a pre-selected, randomized sprint
# location file (see tools/select_training_locations.py -- run ONCE,
# before submitting the array job, to pick this sprint's random training
# subset), runs the generator against exactly those indices, moves the
# output into the shared watch directory, and drops a .ready marker so
# streaming_trainer_watcher.py knows it's safe to process (see that
# script's docstring for the coordination protocol -- .ready is only
# written AFTER the CSV write is complete, so the watcher never races a
# still-being-written file).
#
# Env vars expected (set by the SLURM array script that calls this):
#   RK4TRAIN_DIR         - path to src/RK4TRAIN (already built: bash make.sh)
#   WATCH_DIR           - shared directory the trainer watcher is polling
#   SPRINT_LOCATIONS_FILE - output of select_training_locations.py for this sprint
#   LOCS_PER_TASK       - locations per array task (default 1)
#   SLURM_ARRAY_TASK_ID - set automatically by SLURM

set -euo pipefail

LOCS_PER_TASK="${LOCS_PER_TASK:-1}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set -- run this via sbatch --array, not directly}"
SPRINT_LOCATIONS_FILE="${SPRINT_LOCATIONS_FILE:?SPRINT_LOCATIONS_FILE not set -- run tools/select_training_locations.py first}"

if [ ! -f "$SPRINT_LOCATIONS_FILE" ]; then
    echo "ERROR: $SPRINT_LOCATIONS_FILE not found -- run tools/select_training_locations.py before submitting this array job" >&2
    exit 1
fi

# Extract this task's slice of the pre-selected, randomized location list.
LINE_START=$(( (TASK_ID - 1) * LOCS_PER_TASK + 1 ))
LINE_END=$(( LINE_START + LOCS_PER_TASK - 1 ))
N_TOTAL=$(wc -l < "$SPRINT_LOCATIONS_FILE")
if [ "$LINE_START" -gt "$N_TOTAL" ]; then
    echo "[task $TASK_ID] no locations left for this task index ($LINE_START > $N_TOTAL total) -- exiting cleanly, nothing to do."
    exit 0
fi

echo "[task $TASK_ID] generating locations at lines $LINE_START-$LINE_END of $SPRINT_LOCATIONS_FILE"

cd "$RK4TRAIN_DIR"
mkdir -p "$WATCH_DIR"

# Run in a private scratch subdir keyed by task ID so concurrent array
# tasks on the same node never collide on main's own lattice_batches/
# output naming (main.f90 timestamps its filename, but two tasks starting
# in the same second on the same node could theoretically collide -- this
# sidesteps that entirely).
TASK_SCRATCH="/tmp/rk4tran_task_${SLURM_JOB_ID:-manual}_${TASK_ID}"
mkdir -p "$TASK_SCRATCH"
cp main "$TASK_SCRATCH/"
sed -n "${LINE_START},${LINE_END}p" "$SPRINT_LOCATIONS_FILE" > "$TASK_SCRATCH/task_indices.txt"
cd "$TASK_SCRATCH"

./main --loc-indices-file task_indices.txt

CHUNK_TAG="chunk_task${TASK_ID}"
CSV_FILE=$(ls lattice_batches/*.csv)

# Move (not copy) into the shared watch dir -- move is effectively atomic
# within the same filesystem, avoiding any window where a partial file
# is visible under its final name.
mv "$CSV_FILE" "$WATCH_DIR/${CHUNK_TAG}.csv"
touch "$WATCH_DIR/${CHUNK_TAG}.ready"

echo "[task $TASK_ID] done: $WATCH_DIR/${CHUNK_TAG}.csv (+ .ready marker)"

cd /
rm -rf "$TASK_SCRATCH"
