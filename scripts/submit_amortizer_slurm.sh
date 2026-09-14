#!/bin/bash
#SBATCH --job-name=genepathway-amortizer
#SBATCH --output=logs/%x-%A.out
#SBATCH --error=logs/%x-%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail
mkdir -p logs
source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate genepathway

: "${DATA:?Set DATA to a prior-simulation npz shard from scripts/simulate_prior.py}"
OUTPUT="${OUTPUT:-results/amortizer_${SLURM_JOB_ID}}"

python scripts/train_amortizer.py \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --seed "${SEED:-0}" \
  --epochs "${EPOCHS:-10000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}"
