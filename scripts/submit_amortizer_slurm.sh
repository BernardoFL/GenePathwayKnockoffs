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

: "${DATA:?Set DATA to an NPZ with coordinates, X, L_current, L_prime, F_current, and F_prime}"
OUTPUT="${OUTPUT:-results/amortizer_${SLURM_JOB_ID}}"

python scripts/train_amortizer.py \
  --data "${DATA}" \
  --output "${OUTPUT}" \
  --seed "${SEED:-0}" \
  --H "${H:-8}" \
  --epochs "${EPOCHS:-10000}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}" \
  --graph "${GRAPH:-knn}" \
  --k "${K:-6}"
