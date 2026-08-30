#!/bin/bash
#SBATCH --job-name=bird3-ch-3bx
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
# gambiarra aprovada (15/07): job sem gres (contador untyped do no esta fantasma em 8/8);
# GPUs 0 e 1 fisicamente ociosas e sem alocacao — uso explicito
export CUDA_VISIBLE_DEVICES=0,1
SOLVER=./checkpoints/dr-zero-t2sql/t2sql_solver_bird2b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78
[ -f $SOLVER/config.json ] || { echo "solver hf_model_78 ausente"; exit 1; }
singularity exec --nv \
  --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 \
  --env WANDB_MODE=online --env WANDB_CACHE_DIR=/workspace/tmp/wandb-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache \
  --env CUDA_VISIBLE_DEVICES=0,1 --env NUM_GPUS=2 --env TP=1 \
  --env CH_MICRO=1 --env LOGPROB_MICRO=8 --env CH_EPOCHS=1 \
  --env CHALLENGER_REP_LAMBDA=0.5 \
  --env BASE_MODEL=Qwen/Qwen2.5-Coder-3B-Instruct \
  --env DB_ROOT=./bird/train_databases \
  --env CH_DATA=./data/t2sql_zero_bird.parquet \
  --env VAL_DATA=./data/t2sql_test_bird.parquet \
  --env SAVE_FREQ=25 --env MAX_CKPT=3 \
  /workspace/anon/images/verl0.5-sglang048.sif \
  bash t2sql_challenger.sh bird3_3b $SOLVER data.max_prompt_length=8192
