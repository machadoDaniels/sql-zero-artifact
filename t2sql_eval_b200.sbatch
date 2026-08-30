#!/bin/bash
#SBATCH --job-name=t2sql-eval-b200
#SBATCH --partition=gpu-node-b
#SBATCH --nodes=1
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log

# Avaliação (inferência pura) na B200 usando a imagem vLLM local — a imagem
# verl (cu126) não roda em sm_100, mas p/ servir + avaliar não precisamos dela.
#
# Uso: sbatch t2sql_eval_b200.sbatch <modelo_hf_ou_path> <prefixo_resultado>
#   ex: sbatch t2sql_eval_b200.sbatch anonymous-2321/Think2SQL-3B think2sql3b

set -euo pipefail

MODEL=${1:?modelo}
PREFIX=${2:?prefixo}
RAID=/workspace/anon
PROJECT=/workspace/projects/dr-zero-t2sql
IMAGE=${IMAGE:-${RAID}/images/vllm_openai_nightly.sif}
PORT=$((22000 + ${SLURM_JOB_ID:-0} % 1000))

export HF_HOME=/workspace/hf-cache
mkdir -p "${RAID}/projects/dr-zero-t2sql/logs" "${RAID}/hf-cache"

singularity exec \
    --nv \
    --bind "${RAID}:/workspace" \
    --bind /dev/shm:/dev/shm \
    --pwd "${PROJECT}" \
    --env HF_HOME="${HF_HOME}" \
    "${IMAGE}" \
    bash -c "
set -euo pipefail
python3 -m vllm.entrypoints.openai.api_server \
    --model '${MODEL}' --port ${PORT} --host 127.0.0.1 \
    --download-dir /workspace/hf-cache/hub \
    --gpu-memory-utilization 0.9 --max-model-len 8192 --dtype bfloat16 &
SERVER_PID=\$!
trap 'kill \$SERVER_PID 2>/dev/null || true' EXIT
for _ in \$(seq 1 360); do
    curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null && break
    sleep 5
done
curl -s http://127.0.0.1:${PORT}/v1/models > /dev/null || { echo 'ERRO: vllm não subiu'; exit 1; }

python3 scripts/eval_spider.py \
    --dev_json ./spider/dev.json --db_root ./spider/database \
    --endpoint http://127.0.0.1:${PORT}/v1 --model '${MODEL}' \
    --out ./results/${PREFIX}_spider.json

python3 scripts/eval_spider.py \
    --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases \
    --endpoint http://127.0.0.1:${PORT}/v1 --model '${MODEL}' \
    --exec_timeout 30 --out ./results/${PREFIX}_bird.json
"
echo "eval ${PREFIX} concluído na B200."
