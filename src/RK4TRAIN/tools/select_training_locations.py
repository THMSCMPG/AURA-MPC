#!/usr/bin/env python3
"""select_training_locations.py

Draws a random, reproducible subset of training-location indices from the
~1000-point location pool for one sprint, writing them to a file suitable
for main.f90's --loc-indices-file mode.

Always EXCLUDES the reserved validation indices (see
tools/build_lattice_pools.py's N_VAL_RESERVED -- the first N pool points,
never eligible for training). This is enforced here, not just documented,
so a random draw can never accidentally include a validation location.

Every selection is logged (seed, indices, timestamp) to a running JSONL
file so any given sprint's location set is reproducible after the fact if
you need to debug or re-examine it later -- per the discussion this
implements: "seeded and logged, so any given sprint is reproducible."

Usage:
    python3 select_training_locations.py \\
        --pool-meta ../lattice_pools_generated.f90 \\
        --n-locations 90 \\
        --seed 12345 \\
        --out /tmp/sprint_locations.txt \\
        --log-file selection_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

N_VAL_RESERVED = 12  # must match tools/build_lattice_pools.py's N_VAL_RESERVED


def get_n_locations(pool_f90_path: Path) -> int:
    """Read N_LOCATIONS straight out of the generated Fortran module -- the
    single source of truth for pool size, avoids hardcoding/duplicating it
    here and risking drift if the pool is ever regenerated at a different size."""
    content = pool_f90_path.read_text()
    m = re.search(r"N_LOCATIONS\s*=\s*(\d+)", content)
    if not m:
        raise ValueError(f"Could not find N_LOCATIONS in {pool_f90_path}")
    return int(m.group(1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool-meta", type=Path, required=True, help="Path to lattice_pools_generated.f90 (read-only, just to get N_LOCATIONS)")
    p.add_argument("--n-locations", type=int, required=True, help="How many training locations to select for this sprint")
    p.add_argument("--seed", type=int, default=None, help="RNG seed -- omit for a fresh random seed each call (still logged either way)")
    p.add_argument("--out", type=Path, required=True, help="Output file: one 1-indexed location index per line, for --loc-indices-file")
    p.add_argument("--log-file", type=Path, default=Path("location_selection_history.jsonl"), help="Append-only log of every selection made, for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_total = get_n_locations(args.pool_meta)
    eligible_start = N_VAL_RESERVED + 1  # 1-indexed: pool indices 1..N_VAL_RESERVED are reserved
    n_eligible = n_total - N_VAL_RESERVED

    if args.n_locations > n_eligible:
        raise ValueError(
            f"Requested {args.n_locations} training locations but only {n_eligible} "
            f"are eligible ({n_total} total pool points - {N_VAL_RESERVED} reserved for validation)"
        )

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)
    rng = np.random.default_rng(seed)

    eligible_indices = np.arange(eligible_start, n_total + 1)  # 1-indexed, excludes reserved validation
    selected = rng.choice(eligible_indices, size=args.n_locations, replace=False)
    selected = np.sort(selected)  # sorted for readability; doesn't affect correctness (main.f90 doesn't require sorted input)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for idx in selected:
            f.write(f"{idx}\n")

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n_locations": args.n_locations,
        "n_pool_total": n_total,
        "n_val_reserved": N_VAL_RESERVED,
        "selected_indices": selected.tolist(),
        "out_file": str(args.out),
    }
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"Selected {args.n_locations} training locations (seed={seed}) from {n_eligible} eligible "
          f"({n_total} total, {N_VAL_RESERVED} reserved for validation).")
    print(f"Written to: {args.out}")
    print(f"Logged to: {args.log_file}")


if __name__ == "__main__":
    main()
