#!/bin/bash
#SBATCH --job-name=g3-merge2-rec
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:40:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird2_omni20k_grpo_group8_omnisql-7b
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env SGLANG_ATTENTION_BACKEND=flashinfer /workspace/anon/images/verl0.6-cu128.sif"
$W python -m verl.model_merger merge --backend fsdp --local_dir ./$CKDIR/global_step_$LAST/actor --target_dir ./$CKDIR/hf_model_$LAST
[ -f "$CKDIR/hf_model_$LAST/config.json" ] || { echo "MERGE FALHOU (sem config.json)"; exit 1; }
echo "MERGE2 OK: hf_model_$LAST"
