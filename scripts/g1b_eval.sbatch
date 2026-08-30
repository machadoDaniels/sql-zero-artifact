#!/bin/bash
#SBATCH --job-name=g1b-eval
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird1b20k_grpo_group8_qwen2.5-coder-3b-instruct
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
for N in 75 78; do
  HF=$CKDIR/hf_model_$N
  if [ ! -f "$HF/config.json" ]; then
    $W python -m verl.model_merger merge --backend fsdp \
      --local_dir ./$CKDIR/global_step_$N/actor --target_dir ./$HF
  fi
  [ -f "$HF/config.json" ] || { echo "MERGE FALHOU s$N"; exit 1; }
  [ -f results/g1b20k_s${N}_bird.json ] || $W bash t2sql_baseline.sh ./$HF --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/g1b20k_s${N}_bird.json --concurrency 48
  [ -f results/g1b20k_s${N}_spider.json ] || $W bash t2sql_baseline.sh ./$HF --out ./results/g1b20k_s${N}_spider.json --concurrency 48
  echo "=== g1a s$N avaliado ==="
done
echo "G1A EVAL COMPLETO"
