#!/bin/bash
#SBATCH --job-name=c4-scaling
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
singularity exec --nv \
  --bind /workspace/anon:/workspace \
  --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  --env HF_HOME=/workspace/hf-cache \
  --env PYTHONNOUSERSITE=1 \
  --env SGLANG_ATTENTION_BACKEND=flashinfer \
  --env WANDB_MODE=online \
  --env NUM_GPUS=2 --env TP=1 \
  --env MICRO_BATCH=8 --env LOGPROB_MICRO=16 \
  --env SAVE_FREQ=25 --env MAX_CKPT=12 \
  /workspace/anon/images/verl0.6-cu128.sif \
  bash t2sql_solver.sh ./data/t2sql_solver_iter1.parquet c4scaling
