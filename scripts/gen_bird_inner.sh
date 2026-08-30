#!/bin/bash
set -euo pipefail
PORT=$((24000 + ${SLURM_JOB_ID:-0} % 1000))
CH=./checkpoints/dr-zero-t2sql/t2sql_challenger_bird1_3b_grpo_batch_group1-5_qwen2.5-coder-3b-instruct/hf_model
python -m sglang.launch_server --model=$CH --port=$PORT --mem-fraction-static=0.85 --log-level=error &
SP=$!
trap "kill $SP 2>/dev/null || true" EXIT
for _ in $(seq 1 240); do curl -s http://127.0.0.1:$PORT/health >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:$PORT/health >/dev/null || { echo "ERRO: sglang nao subiu"; exit 1; }
python t2sql_data.py --sample_rows 2 gen-solver-data --endpoint http://127.0.0.1:$PORT/v1 --model $CH --db_root ./bird/train_databases --db_list ./bird/train_dbs_keep.txt --num_prompts 20000 --samples_per_prompt 5 --max_pairs 20000 --out ./data/t2sql_solver_bird1.parquet --concurrency 64
