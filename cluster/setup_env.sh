#!/bin/bash
# One-time setup on a cluster login node: create .venv/ with the CPU build of
# PyTorch plus requirements.txt.
#
#     bash cluster/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load miniconda 2>/dev/null || module load Python 2>/dev/null || true
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" --version

if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel
# CPU-only wheel: the array jobs never use a GPU.
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0"
python -m pip install -r requirements.txt

python - <<'PY'
import torch, gpytorch
print(f"torch {torch.__version__}  gpytorch {gpytorch.__version__}  threads {torch.get_num_threads()}")
PY
