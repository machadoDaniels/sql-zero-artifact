#!/bin/bash
#SBATCH --job-name=g1a-ch-bird
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
singularity exec --nv   --bind /workspace/anon:/workspace   --bind /dev/shm:/dev/shm   --pwd /workspace/projects/dr-zero-t2sql   --env HF_HOME=/workspace/hf-cache   --env PYTHONNOUSERSITE=1   --env SGLANG_ATTENTION_BACKEND=flashinfer   --env WANDB_MODE=online   --env NUM_GPUS=2 --env TP=1   --env CH_MICRO=2 --env LOGPROB_MICRO=8   --env CH_EPOCHS=1   --env CHALLENGER_REP_LAMBDA=0.5   --env DB_ROOT=./bird/train_databases   --env CH_DATA=./data/t2sql_zero_bird.parquet   --env VAL_DATA=./data/t2sql_test_bird.parquet   --env SAVE_FREQ=25 --env MAX_CKPT=3   /workspace/anon/images/verl0.6-cu128.sif   bash t2sql_challenger.sh bird1_3b Qwen/Qwen2.5-Coder-3B-Instruct data.max_prompt_length=8192
