#!/bin/bash
#SBATCH --job-name=bird3-ch-7b
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
singularity exec --nv \
  --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 \
  --env SGLANG_ATTENTION_BACKEND=flashinfer \
  --env WANDB_MODE=online --env WANDB_CACHE_DIR=/workspace/tmp/wandb-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache \
  --env NUM_GPUS=2 --env TP=1 \
  --env CH_MICRO=1 --env LOGPROB_MICRO=8 --env CH_EPOCHS=1 \
  --env CHALLENGER_REP_LAMBDA=0.5 \
  --env BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
  --env DB_ROOT=./bird/all_databases \
  --env CH_DATA=./data/t2sql_zero_bird.parquet \
  --env VAL_DATA=./data/t2sql_test_bird.parquet \
  --env SAVE_FREQ=25 --env MAX_CKPT=3 \
  /workspace/anon/images/verl0.6-cu128.sif \
  bash t2sql_challenger.sh bird3_7b ./checkpoints/dr-zero-t2sql/t2sql_solver_bird2_7b20k_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_78 data.max_prompt_length=8192
