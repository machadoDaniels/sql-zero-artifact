#!/bin/bash
#SBATCH --job-name=f5-test-0g
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=6
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env CUDA_VISIBLE_DEVICES=6 --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
$W bash t2sql_baseline.sh anonymous/sql-zero-7b-solver-omnisql --dev_json ./spider/test.json --db_root ./spider/test_database --out ./results/f5_s200_spidertest.json --concurrency 48
echo COMPLETO
