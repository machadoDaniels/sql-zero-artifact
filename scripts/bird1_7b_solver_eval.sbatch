#!/bin/bash
#SBATCH --job-name=bird1-solver-7b
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
[ -f data/t2sql_solver_bird1_7b_20k.parquet ] || { echo "gen ausente"; exit 1; }
singularity exec --nv \
  --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 \
  --env WANDB_MODE=online --env WANDB_CACHE_DIR=/workspace/tmp/wandb-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache \
  --env NUM_GPUS=2 --env TP=1 \
  --env SOLVER_GPU=2 --env TRAIN_GPUS=0,1 \
  --env MICRO_BATCH=1 --env LOGPROB_MICRO=4 --env ROLLOUT_MEM=0.15 \
  --env BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
  --env DB_ROOT=./bird/all_databases \
  --env VAL_DATA=./data/t2sql_test_bird.parquet \
  --env SAVE_FREQ=25 --env MAX_CKPT=6 \
  --env SOLVER_THINK_BONUS=0.1 \
  /workspace/anon/images/verl0.5-sglang048.sif \
  bash t2sql_solver.sh ./data/t2sql_solver_bird1_7b_20k.parquet bird1_7b20k data.max_prompt_length=8192 actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8704 actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird1_7b20k_grpo_group8_qwen2.5-coder-7b-instruct
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
$W python -m verl.model_merger merge --backend fsdp --local_dir ./$CKDIR/global_step_$LAST/actor --target_dir ./$CKDIR/hf_model_$LAST
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/bird1_7b20k_s${LAST}_bird.json --concurrency 48
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --out ./results/bird1_7b20k_s${LAST}_spider.json --concurrency 48
echo "BIRD1-7B COMPLETO"
