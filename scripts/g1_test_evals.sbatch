#!/bin/bash
#SBATCH --job-name=g1-test-evals
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
[ -f results/g1b20k_s78_spidertest.json ] || $W bash t2sql_baseline.sh ./checkpoints/dr-zero-t2sql/t2sql_solver_bird1b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78 --dev_json ./spider/test.json --db_root ./spider/test_database --out ./results/g1b20k_s78_spidertest.json --concurrency 48
[ -f results/g1a_s75_spidertest.json ] || $W bash t2sql_baseline.sh anonymous/sql-zero-3b-solver-bird1a --dev_json ./spider/test.json --db_root ./spider/test_database --out ./results/g1a_s75_spidertest.json --concurrency 48
echo "TEST EVALS COMPLETOS"
