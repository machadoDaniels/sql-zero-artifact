#!/bin/bash
#SBATCH --job-name=t2sql-solver
#SBATCH --partition=gpu-node-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log

# Standalone solver GRPO on an arbitrary dataset (e.g. the gold-train control).
# Usage: sbatch t2sql_solver.sbatch <train_parquet> <tag> [SOLVER_INIT overrides via env]

set -euo pipefail
TRAIN=${1:?train parquet}
TAG=${2:?tag}
RAID=/workspace/anon

# Slurm exports ROCR/HIP_VISIBLE_DEVICES (AMD) alongside CUDA; verl rejects it.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

singularity exec \
    --nv \
    --bind "${RAID}:/workspace" \
    --bind /dev/shm:/dev/shm \
    --pwd /workspace/projects/dr-zero-t2sql \
    --env HF_HOME=/workspace/hf-cache \
    --env WANDB_MODE="${WANDB_MODE:-online}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env NUM_GPUS="${NUM_GPUS:-2}" \
    --env TP="${TP:-1}" \
    --env SAVE_FREQ="${SAVE_FREQ:-25}" \
    --env MAX_CKPT="${MAX_CKPT:-2}" \
    --env SOLVER_INIT="${SOLVER_INIT:-Qwen/Qwen2.5-Coder-3B-Instruct}" \
    --env MODEL_NICK="${MODEL_NICK:-qwen2.5-coder-3b-instruct}" \
    --env SOLVER_THINK_BONUS="${SOLVER_THINK_BONUS:-0.0}" \
    --env DB_ROOT="${DB_ROOT:-}" \
    --env CH_DATA="${CH_DATA:-}" \
    --env VAL_DATA="${VAL_DATA:-}" \
    --env MICRO_BATCH="${MICRO_BATCH:-8}" \
    --env LOGPROB_MICRO="${LOGPROB_MICRO:-16}" \
    --env ROLLOUT_MEM="${ROLLOUT_MEM:-0.4}" \
    "${RAID}/images/verl0.5-sglang048.sif" \
    bash t2sql_solver.sh "${TRAIN}" "${TAG}"
