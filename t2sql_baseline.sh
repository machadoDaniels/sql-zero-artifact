# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Passo 0 (POC): zero-shot baseline on Spider dev — the number every
# self-play iteration must beat (docs/DECISOES_ARQUITETURA.md §6).
#
# Usage: bash t2sql_baseline.sh [model] [extra eval_spider.py args...]

set -x

model=${1:-Qwen/Qwen2.5-Coder-3B-Instruct}
if [ $# -ge 1 ]; then shift; fi
model_name=$(basename "$model" | tr '[:upper:]' '[:lower:]')
port=${PORT:-$((20000 + $(( ${SLURM_JOB_ID:-0} % 1000 ))))}

source activate llm 2>/dev/null || true;  # no container (singularity) o env ja e o do sistema


# Espera um endpoint de health responder; falha alto em vez de travar p/ sempre.
wait_health() {
    for _ in $(seq 1 360); do
        curl -s "$1" > /dev/null && return 0
        sleep 5
    done
    echo "ERRO: timeout (30min) esperando $1 — servidor não subiu."
    exit 1
}

python -m sglang.launch_server \
    --model=${model} \
    --port=${port} \
    --log-level=error &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

wait_health http://127.0.0.1:${port}/health

python scripts/eval_spider.py \
    --dev_json ./spider/dev.json \
    --db_root ./spider/database \
    --endpoint http://127.0.0.1:${port}/v1 \
    --model ${model} \
    --out ./results/baseline_${model_name}.json $@

kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
