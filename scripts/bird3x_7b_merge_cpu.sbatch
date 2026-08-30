#!/bin/bash
#SBATCH --job-name=bird3x-7b-merge-cpu
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird3x_7b20k_grpo_group8_qwen2.5-coder-7b-instruct
W="singularity exec --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env SGLANG_ATTENTION_BACKEND=flashinfer /workspace/anon/images/verl0.6-cu128.sif"
$W python -m verl.model_merger merge --backend fsdp --local_dir ./$CKDIR/global_step_78/actor --target_dir ./$CKDIR/hf_model_78
[ -f ./$CKDIR/hf_model_78/config.json ] || { echo MERGE-FALHOU; exit 1; }
echo BIRD3X-MERGE-CPU-COMPLETO
