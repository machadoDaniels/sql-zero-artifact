#!/bin/bash
# Cadeia de evals docker, 1 GPU: cada item "model[@rev]|dev_json|db_root|out"
# uso: h100n1_docker_eval_chain.sh <GPU_ID> <nome> <item1> [item2...]
set -uo pipefail
GPU=$1; NAME=$2; shift 2
ROOT=/workspace/anon
PORT=$((23000 + GPU))
for item in "$@"; do
  IFS="|" read -r modelrev dev db out <<< "$item"
  model=${modelrev%%@*}; rev=""
  [[ "$modelrev" == *@* ]] && rev="--revision ${modelrev##*@}"
  echo "=== $modelrev -> $out ==="
  docker run --rm --name "${NAME}" --gpus "\"device=$GPU\"" \
    -v $ROOT:/workspace -w /workspace/projects/dr-zero-t2sql \
    --env HF_HOME=/workspace/hf-cache --env HF_TOKEN=$(cat $ROOT/hf-cache/token) \
    --shm-size=16g --entrypoint bash vllm/vllm-openai:v0.23.0 -c "
vllm serve $model $rev --port $PORT --gpu-memory-utilization 0.75 > /tmp/vs.log 2>&1 &
SPID=\$!
for i in \$(seq 1 360); do curl -s http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
curl -s http://127.0.0.1:$PORT/health > /dev/null || { echo SERVER-TIMEOUT; tail -20 /tmp/vs.log; exit 1; }
python3 scripts/eval_spider.py --dev_json $dev --db_root $db --endpoint http://127.0.0.1:$PORT/v1 --model $model --out $out --concurrency 48
kill \$SPID 2>/dev/null || true" || echo "FALHOU: $modelrev"
done
echo CADEIA-COMPLETA
