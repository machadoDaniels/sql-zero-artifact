#!/bin/bash
#SBATCH --job-name=g3-eval2-rec
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird2_omni20k_grpo_group8_omnisql-7b
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
[ -f "$CKDIR/hf_model_$LAST/config.json" ] || { echo "hf_model ausente"; exit 1; }
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env SGLANG_ATTENTION_BACKEND=flashinfer --env SLURM_JOB_ID=$SLURM_JOB_ID /workspace/anon/images/verl0.6-cu128.sif"
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/bird2_omni20k_s${LAST}_bird.json --concurrency 48
$W bash t2sql_baseline.sh ./$CKDIR/hf_model_$LAST --out ./results/bird2_omni20k_s${LAST}_spider.json --concurrency 48
echo "EVAL2 OK"
