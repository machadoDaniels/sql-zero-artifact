#!/bin/bash
#SBATCH --job-name=bird2-kfilter
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
PORT=$((24000 + SLURM_JOB_ID % 1000))
EPORT=$((25000 + SLURM_JOB_ID % 1000))
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache /workspace/anon/images/verl0.5-sglang048.sif"
$W bash -c "
python search/sql_exec_server.py --db_root ./bird/all_databases --port $EPORT &
EP=\$!
python -m sglang.launch_server --model=./checkpoints/dr-zero-t2sql/t2sql_solver_bird1b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78 --port=$PORT --mem-fraction-static=0.85 --log-level=error &
SP=\$!
trap \"kill \$SP \$EP 2>/dev/null || true\" EXIT
for _ in \$(seq 1 240); do curl -s http://127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
for _ in \$(seq 1 60); do curl -s http://127.0.0.1:$EPORT/health >/dev/null && break; sleep 2; done
python scripts/score_gen_k.py --parquet ./data/t2sql_solver_bird2_20k.parquet --out ./data/t2sql_solver_bird2_kfiltered.parquet --endpoint http://127.0.0.1:$PORT/v1 --model ./checkpoints/dr-zero-t2sql/t2sql_solver_bird1b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78 --exec_url http://127.0.0.1:$EPORT --n 5 --concurrency 48
"
echo KFILTER-COMPLETO
