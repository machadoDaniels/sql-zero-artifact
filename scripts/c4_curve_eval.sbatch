#!/bin/bash
#SBATCH --job-name=c4-curve-eval
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_c4scaling_grpo_group8_qwen2.5-coder-3b-instruct
W6="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.6-cu128.sif"
W5="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
for N in 25 50 75 100 125 150 175 200 225 226; do
  HF=$CKDIR/hf_model_$N
  if [ ! -f "$HF/config.json" ]; then
    $W6 python -m verl.model_merger merge --backend fsdp \
      --local_dir ./$CKDIR/global_step_$N/actor --target_dir ./$HF
  fi
  [ -f "$HF/config.json" ] || { echo "MERGE FALHOU s$N"; exit 1; }
  [ -f results/c4_s${N}_spider.json ] || $W5 bash t2sql_baseline.sh ./$HF --out ./results/c4_s${N}_spider.json --concurrency 48
  [ -f results/c4_s${N}_bird.json ] || $W5 bash t2sql_baseline.sh ./$HF --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out ./results/c4_s${N}_bird.json --concurrency 48
  echo "=== ponto s$N concluido ==="
done
echo "CURVA COMPLETA"
