#!/usr/bin/env bash
# workstation/run_workstation.sh – Convenience launcher for the AURA workstation
# inference server.
#
# Sources ROOT paths from the bootstrap convention established by
# PIED-AURA-MFP/bootstrap.sh and passes all extra arguments to
# workstation.cli.
#
# Usage:
#   ./workstation/run_workstation.sh [--port 8765] [--optimiser pso] ...
#
# Required environment variables (or their defaults):
#   ROOT            – aura-stack root directory (default: $HOME/aura-stack)
#   AURA_ROOT       – AURA-MFP repository path  (default: $ROOT/AURA-MFP)
#   PINN_ROOT       – PINN-AURA-MFP path         (default: $ROOT/PINN-AURA-MFP)
#   PINN_CHECKPOINT – checkpoint file             (default: $PINN_ROOT/checkpoints/best.pt)

set -euo pipefail

ROOT="${ROOT:-$HOME/aura-stack}"
AURA_ROOT="${AURA_ROOT:-$ROOT/AURA-MFP}"
PINN_ROOT="${PINN_ROOT:-$ROOT/PINN-AURA-MFP}"
PINN_CHECKPOINT="${PINN_CHECKPOINT:-$PINN_ROOT/checkpoints/best.pt}"

VENV="$ROOT/.venv"

if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
fi

exec python -m workstation.cli \
    --aura-root  "$AURA_ROOT" \
    --pinn-root  "$PINN_ROOT" \
    --checkpoint "$PINN_CHECKPOINT" \
    "$@"
