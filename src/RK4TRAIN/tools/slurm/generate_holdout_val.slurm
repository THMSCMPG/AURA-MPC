#!/bin/bash
#SBATCH --job-name=rk4tran_val
#SBATCH --output=logs/val_%j.out
#SBATCH --error=logs/val_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
##SBATCH --partition=PARTITION_NAME_HERE   # <-- fill in, see generate_chunks.slurm

# Generates the held-out validation set ONCE, before the training array job
# and watcher start. Run this FIRST, wait for it to finish, THEN run
# tools/select_training_locations.py and submit generate_chunks.slurm +
# train_watcher.slurm.
#
# Uses pool indices 1-12 -- the RESERVED validation indices (see
# tools/build_lattice_pools.py's N_VAL_RESERVED), a fixed, well-distributed
# subset of the ~1050-point location pool that's permanently excluded from
# random per-sprint training selection (tools/select_training_locations.py
# enforces this by construction, not just by convention). These stay fixed
# across the ENTIRE multi-sprint campaign -- unlike training locations,
# which are randomized fresh each sprint, you want one stable, comparable
# validation metric to actually tell whether sprint 40 is better than
# sprint 5.

set -euo pipefail

export RK4TRAIN_DIR="${RK4TRAIN_DIR:-$HOME/AURA-MPC/src/RK4TRAIN}"
export RUN_DIR="${RUN_DIR:-$HOME/aura_mpc_run}"

mkdir -p "$RUN_DIR/memmaps" logs

cd "$RK4TRAIN_DIR"
./main --loc-range 1 12
CSV_FILE=$(ls -t lattice_batches/*.csv | head -1)

cd "$(dirname "$0")/../.."   # repo root (tools/slurm/../..)
python3 -c "
import sys
sys.path.insert(0, 'src/RK4TRAIN/ml')
from data.memmap_dataset import build_memmap
from pathlib import Path
build_memmap([Path('$RK4TRAIN_DIR/$CSV_FILE')], Path('$RUN_DIR/memmaps'), name='holdout_val')
"

echo "Held-out validation set ready: $RUN_DIR/memmaps/holdout_val.meta.json"
echo "Now run tools/select_training_locations.py, then submit generate_chunks.slurm and train_watcher.slurm."
