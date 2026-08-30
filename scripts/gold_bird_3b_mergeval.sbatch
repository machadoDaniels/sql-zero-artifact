#!/bin/bash
#SBATCH --job-name=goldbird3b-mergeval
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=6:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_goldbird3b_grpo_group8_qwen2.5-coder-3b-instruct
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.5-sglang048.sif"
[ -f $CKDIR/hf_model_$LAST/config.json ] || $W python -m verl.model_merger merge --backend fsdp --local_dir ./$CKDIR/global_step_$LAST/actor --target_dir ./$CKDIR/hf_model_$LAST
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/goldbird3b_s${LAST}_bird.json --concurrency 48
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --out ./results/goldbird3b_s${LAST}_spider.json --concurrency 48
echo "GOLD-BIRD-3B MERGEVAL COMPLETO"
