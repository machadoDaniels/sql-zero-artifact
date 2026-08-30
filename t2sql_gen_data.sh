# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Generate + filter solver training data with a served proposer model.
#   iter0: proposer = base model (no RL) -> validates the whole loop cheaply
#   iter1+: proposer = converted challenger checkpoint
#
# Usage: bash t2sql_gen_data.sh [proposer_model_or_ckpt] [out_parquet] [num_prompts]

set -x

proposer=${1:-Qwen/Qwen2.5-Coder-3B-Instruct}
out=${2:-./data/t2sql_solver_iter0.parquet}
num_prompts=${3:-4000}
port=${PORT:-$((21000 + $(( ${SLURM_JOB_ID:-0} % 1000 ))))}

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
    --model=${proposer} \
    --port=${port} \
    --log-level=error &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

wait_health http://127.0.0.1:${port}/health

python t2sql_data.py --spider_dir ./spider gen-solver-data \
    --endpoint http://127.0.0.1:${port}/v1 \
    --model ${proposer} \
    --out ${out} \
    --num_prompts ${num_prompts} \
    --samples_per_prompt 5

kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true
