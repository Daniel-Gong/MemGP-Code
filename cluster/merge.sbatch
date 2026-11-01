#!/bin/bash
# Merge the shards of one stage.  STAGE=1|2 (required).
#SBATCH --job-name=memgp_merge
#SBATCH --output=cluster/logs/%x_%A.out
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00

source cluster/activate.sh
python scripts/run_tasks.py --merge "${STAGE:?set STAGE=1|2}"
