#!/bin/bash
# Sourced by every sbatch script: cd to the repository root and activate the
# Python environment. Override the environment with MEMGP_VENV=/path/to/venv.
set -euo pipefail

REPO_ROOT="${MEMGP_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module load miniconda 2>/dev/null || module load Python 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "${MEMGP_VENV:-$REPO_ROOT/.venv}/bin/activate"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

echo "[activate] host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-NA}" \
     "cpus=${SLURM_CPUS_PER_TASK:-NA} repo=$REPO_ROOT python=$(command -v python)"
