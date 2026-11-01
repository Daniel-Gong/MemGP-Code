#!/bin/bash
# Run every simulation task with two subjects and a trial or two each into a
# scratch directory, merge, and render all figures from that output. Checks
# the installation and the whole pipeline in about 15 minutes on a laptop; the
# numbers are meaningless at this size. results/ is not touched.
#
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/memgp_smoke.XXXXXX")"
export MEMGP_RESULTS="$SCRATCH/results"
export MEMGP_FIGURES="$SCRATCH/figures"
mkdir -p "$MEMGP_RESULTS/bias"
cp results/bias/adaptive_bias_locked_design.json "$MEMGP_RESULTS/bias/"
echo "Scratch output: $SCRATCH"

"$PYTHON" scripts/run_tasks.py --smoke
PYTHON="$PYTHON" bash scripts/make_figures.sh
echo
echo "Smoke test passed. Output in $SCRATCH"
