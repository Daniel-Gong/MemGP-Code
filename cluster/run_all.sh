#!/bin/bash
# Submit every manuscript simulation to Slurm: the stage-1 array, its merge,
# the stage-2 array (per-subject model fits) and its merge, chained with
# dependencies.
#
#   export MEMGP_ACCOUNT=<slurm account> MEMGP_PARTITION=<cpu partition>
#   bash cluster/run_all.sh
#
# MEMGP_DRY_RUN=1 prints the sbatch commands without submitting.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export MEMGP_REPO="$REPO_ROOT"
source cluster/activate.sh >/dev/null

ACCOUNT="${MEMGP_ACCOUNT:?set MEMGP_ACCOUNT}"
PARTITION="${MEMGP_PARTITION:?set MEMGP_PARTITION}"
DRY="${MEMGP_DRY_RUN:-0}"
mkdir -p cluster/logs

N1=$(python scripts/run_tasks.py --stage 1 --count)
N2=$(python scripts/run_tasks.py --stage 2 --count)

submit() {   # submit <dependency or ""> <sbatch args...>
    local dep="$1"; shift
    local cmd=(sbatch --parsable -A "$ACCOUNT" -p "$PARTITION")
    [ -n "$dep" ] && cmd+=(--dependency="afterok:$dep")
    cmd+=("$@")
    if [ "$DRY" = "1" ]; then echo "DRY: ${cmd[*]}" >&2; echo 0; return; fi
    "${cmd[@]}"
}

J1=$(submit "" --array="0-$((N1 - 1))" --export=ALL,STAGE=1 cluster/task.sbatch)
M1=$(submit "$J1" --export=ALL,STAGE=1 cluster/merge.sbatch)
J2=$(submit "$M1" --array="0-$((N2 - 1))" --export=ALL,STAGE=2 cluster/task.sbatch)
M2=$(submit "$J2" --export=ALL,STAGE=2 cluster/merge.sbatch)
echo "stage 1: $N1 tasks (job $J1), merge $M1; stage 2: $N2 tasks (job $J2), merge $M2"
echo "Monitor with: squeue --me"
