#!/bin/bash
#SBATCH --job-name=g3-solver-omni-h2
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
[ -f data/t2sql_solver_bird1_omni_20k.parquet ] || { echo "gen ausente"; exit 1; }
singularity exec --nv   --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm   --pwd /workspace/projects/dr-zero-t2sql   --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1   --env WANDB_MODE=online --env WANDB_CACHE_DIR=/workspace/tmp/wandb-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache   --env NUM_GPUS=4 --env TP=1   --env MICRO_BATCH=1 --env LOGPROB_MICRO=8 --env ROLLOUT_MEM=0.2   --env BASE_MODEL=seeklhy/OmniSQL-7B   --env DB_ROOT=./bird/all_databases   --env VAL_DATA=./data/t2sql_test_bird.parquet   --env SAVE_FREQ=25 --env MAX_CKPT=1   --env SOLVER_THINK_BONUS=0.1   /workspace/anon/images/verl0.5-sglang048.sif   bash t2sql_solver.sh ./data/t2sql_solver_bird1_omni_20k.parquet bird1_omni20k data.max_prompt_length=8192
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird1_omni20k_grpo_group8_omnisql-7b
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.5-sglang048.sif"
$W python -m verl.model_merger merge --backend fsdp --local_dir ./$CKDIR/global_step_$LAST/actor --target_dir ./$CKDIR/hf_model_$LAST
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/bird1_omni20k_s${LAST}_bird.json --concurrency 48
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --out ./results/bird1_omni20k_s${LAST}_spider.json --concurrency 48
echo "G3-ITER1-H100N2 COMPLETO"
