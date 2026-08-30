#!/bin/bash
set -euo pipefail
PORT=$((23000 + ${SLURM_JOB_ID:-0} % 1000))
MODEL=./checkpoints/dr-zero-t2sql/t2sql_solver_omnisql_grpo_group8_omnisql-7b/hf_model_200
python3 -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $PORT --host 127.0.0.1 --gpu-memory-utilization 0.9 --max-model-len 8192 --dtype bfloat16 &
SP=$!
trap "kill $SP 2>/dev/null || true" EXIT
for _ in $(seq 1 240); do curl -s http://127.0.0.1:$PORT/v1/models >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:$PORT/v1/models >/dev/null || { echo "ERRO: vllm nao subiu"; exit 1; }
python3 scripts/eval_spider.py --dev_json ./robustness/spider_syn_dev.json --db_root ./spider/database --endpoint http://127.0.0.1:$PORT/v1 --model "$MODEL" --out ./results/syn_omnisql_s200.json --concurrency 48
