#!/bin/bash
#SBATCH --job-name=g1-solver
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
  --env MICRO_BATCH=4 --env LOGPROB_MICRO=8 \
  --env DB_ROOT=./bird/all_databases \
  --env VAL_DATA=./data/t2sql_test_bird.parquet \
  --env SAVE_FREQ=25 --env MAX_CKPT=6 \
  --env SOLVER_THINK_BONUS=${THINK_BONUS:-0.0} \
  /workspace/anon/images/verl0.6-cu128.sif \
  bash t2sql_solver.sh ./data/t2sql_solver_bird1.parquet ${TAG:-bird1a} data.max_prompt_length=8192
