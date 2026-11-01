#!/bin/bash
# One task of experiments/registry.py per array task.
#   STAGE=1|2   (required)
#
#   sbatch -A <acct> -p <part> --array=0-$((N-1)) --export=ALL,STAGE=1 cluster/task.sbatch
# where N = python scripts/run_tasks.py --stage 1 --count
#SBATCH --job-name=memgp_task
#SBATCH --output=cluster/logs/%x_%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G
#SBATCH --time=06:00:00

source cluster/activate.sh
export TORCH_NUM_THREADS=1
python scripts/run_tasks.py --stage "${STAGE:?set STAGE=1|2}" --index "${SLURM_ARRAY_TASK_ID:?}"
