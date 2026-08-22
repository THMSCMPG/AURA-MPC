# Running the streaming pipeline on Loki (SLURM)

## Real throughput numbers (measured, not guessed — worth knowing before you tune anything)

Earlier advice in this project said "prioritize generator nodes over trainer
nodes" — that was based on an assumption, not a measurement, and it turned
out to be incomplete. When actually benchmarked at realistic scale (1.76M
real rows, not a toy sample), training didn't even finish within several
minutes. The cause wasn't batch size or CPU core count — it was a genuine
software bug: the dataset's row-access path built 4-6 separate small
tensors per row in a Python loop, an easy trap for tabular PyTorch data.
Fixed by adding a vectorized batch loader (`RK4TRANMemmapDataset.batch_iterator`)
that reads and converts an entire batch in one numpy operation instead of
looping per-row. Measured speedup: **58x** on data loading alone.

With that fixed, the real picture is close to your own original instinct:
at `--epochs-per-chunk=1` (the new default), training runs at **~0.95x
generation's own pace** — almost exactly matched, not "way slower" or "way
faster." That's why `generate_chunks.slurm` now defaults to 3 concurrent
generators feeding 1 trainer (matching your 4-node cap) rather than a much
larger generator pool — beyond keeping the single trainer's queue fed, more
generators don't increase total throughput, they just build up backlog the
trainer can't consume any faster.

If you raise `--epochs-per-chunk` above 1, training slows roughly linearly
(2 epochs ≈ 1.9x generation pace, 3 epochs ≈ 2.8x) and the trainer becomes
the real bottleneck regardless of how many generator nodes you throw at it,
since only one trainer can safely exist (concurrent checkpoint writers
would corrupt the shared model file).


Three jobs, run in this order:

## 1. Generate the held-out validation set (once, wait for it to finish)

```bash
sbatch tools/slurm/generate_holdout_val.slurm
squeue -u $USER          # wait for it to complete
```

Uses pool indices 1-12 -- a fixed, well-distributed, permanently-reserved
subset of the ~1050-point location pool (see `tools/build_lattice_pools.py`'s
`N_VAL_RESERVED`). These stay the same for the entire multi-sprint
campaign, so validation results are actually comparable sprint-to-sprint.

## 2. Pick this sprint's random training locations

```bash
python3 tools/select_training_locations.py \
    --pool-meta src/RK4TRAIN/lattice_pools_generated.f90 \
    --n-locations 90 \
    --out "$RUN_DIR/sprint_locations.txt" \
    --log-file "$RUN_DIR/location_selection_history.jsonl"
```

Instant (no Fortran involved), so this runs directly, no `sbatch` needed.
Draws a random 90-location subset from the ~1038 eligible pool points
(excludes the 12 reserved validation indices by construction, not just by
convention). Every selection is logged with its seed to
`location_selection_history.jsonl`, so any sprint's exact location set is
reproducible later if you need to debug it. Omit `--seed` for a fresh
random draw each time (still logged) -- this is what gives you the
"different data next time you run it" property, without needing live
elevation lookups per sprint (the whole pool's elevation was already
fetched once, up front).

## 3. Submit generation (array job) and training (single persistent job) together

```bash
sbatch tools/slurm/generate_chunks.slurm
sbatch tools/slurm/train_watcher.slurm
```

These two run concurrently. The array job produces location-chunks in
parallel across Loki's nodes (each array task is single-core, so SLURM can
pack up to 8 per node — adjust `--array=1-90%N` in `generate_chunks.slurm`
if your allocation gives you a different effective concurrency). The
watcher job is the single process that owns the model checkpoint, polling
the shared directory both jobs point at and consuming chunks as they
appear, one at a time, in whatever order they finish (not necessarily
location order).

## 4. Monitor progress

```bash
tail -f logs/train_*.out                                   # training progress, val_loss per chunk
cat $RUN_DIR/checkpoints/streaming_metrics.csv              # full history
squeue -u $USER                                             # job status
```

## 5. After everything finishes: render the plots

```bash
python3 tools/merge_chunk_aggregates.py \
    --raw-dir $RUN_DIR/raw_aggregates \
    --out-dir $RUN_DIR/plots
cd $RUN_DIR/plots && gnuplot /path/to/repo/tools/plot_synthetic_data.gp
```

The merge step is safe to run *at any point*, not just at the end — it
just recomputes the final means/stddevs from whatever raw accumulator
files exist so far, so you can check in on partial results mid-sprint
without disturbing anything still running.

---

## Before any of this: fill in the blanks

- **Partition name** — every `.slurm` file has a commented-out
  `#SBATCH --partition=PARTITION_NAME_HERE` line. Run `sinfo` to see
  Loki's actual partition names, then uncomment and fill in.
- **`RUN_DIR`** and **`RK4TRAIN_DIR`** — set as environment variables before
  submitting, or edit the defaults at the top of each `.slurm` file. Both
  need to be on Loki's shared NFS filesystem (anywhere under your home
  directory works) so all array tasks and the watcher can see the same
  files.
- **Build RK4TRAN first**: `cd src/RK4TRAIN && bash make.sh` — do this once,
  before submitting anything. The array job assumes `main` already exists.

## What I know about Loki vs. what I don't

I found real info from APSU's physics department page: 32 compute nodes,
8 cores each (256 total), 16GB RAM/node (4 nodes at 32GB), gigabit
ethernet, shared NFS filesystem (26TB on the head node). I don't know for
certain this is the exact allocation you have access to, whether the specs
have changed since that page was last updated, or the actual SLURM
partition names/QOS/walltime limits — those aren't public information and
need `sinfo`/`sacctmgr` output from your own account. The scripts are
written to be easy to adjust (array concurrency, memory/time requests,
locations-per-task) once you have those specifics.

## Why two separate jobs instead of one

Generation is embarrassingly parallel (every location is fully
independent) but training isn't — only one process can safely own the
persistent model checkpoint at a time, or concurrent writes would corrupt
it. Splitting into a parallel array (generation) + one funnel process
(training) gets you the parallelism where it's actually available without
needing distributed training infrastructure this cluster doesn't have
(no GPUs, gigabit not high-speed interconnect — not a good fit for tightly
-coupled multi-node training anyway).
