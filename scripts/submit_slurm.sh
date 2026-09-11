#!/bin/bash
#SBATCH --job-name=genepathway-mcmc
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7

set -euo pipefail
mkdir -p logs
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate genepathway

DATA="${DATA:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/${SLURM_JOB_ID}}"
ARGS=(--output "${OUTPUT_ROOT}/seed_${SLURM_ARRAY_TASK_ID}" --seed "${SLURM_ARRAY_TASK_ID}" --iterations "${ITERATIONS:-10000}" --warmup "${WARMUP:-2000}")
if [[ -n "${DATA}" ]]; then
  ARGS+=(--data "${DATA}")
fi

python scripts/run_hpc.py "${ARGS[@]}"