#!/bin/bash
#SBATCH --job-name=t2sql-poc
#SBATCH --partition=gpu-node-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log

# POC do Dr. Zero -> Text-to-SQL via singularity (default: gpu-node-a, 3x H100).
# Para B200 ajuste partition/gres e use uma imagem cu128 (ver README do job).
#
# Uso:
#   sbatch t2sql_poc.sbatch            # tudo: prep -> baseline -> gen0 -> solver0
#   sbatch t2sql_poc.sbatch baseline   # um estágio específico
#   SMOKE=1 sbatch t2sql_poc.sbatch    # smoke test antes do treino real
#
# Imagem (uma vez, no nó com internet):
#   singularity pull /workspace/anon/images/verl0.5-sglang048.sif \
#       docker://verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2
#
# Pré-requisito de dados (uma vez): spider_data.zip de
# https://yale-lily.github.io/spider extraído via scripts/download_spider.py

set -euo pipefail

RAID=/workspace/anon
PROJECT_HOST=${RAID}/projects/dr-zero-t2sql
PROJECT_CTN=/workspace/projects/dr-zero-t2sql   # visão de dentro do container
IMAGE=${IMAGE:-${RAID}/images/verl0.5-sglang048.sif}

STAGE=${1:-all}

export HF_HOME=/workspace/hf-cache
export OMP_NUM_THREADS=8
export WANDB_MODE=${WANDB_MODE:-online}   # login persistente via ~/.netrc no nó; use offline se a rede falhar

mkdir -p "${PROJECT_HOST}/logs" "${RAID}/hf-cache"

if [ ! -f "${IMAGE}" ]; then
    echo "ERRO: imagem não encontrada: ${IMAGE}"
    echo "Baixe com:"
    echo "  singularity pull ${IMAGE} docker://verlai/verl:app-verl0.5-sglang0.4.8-mcore0.12.2-te2.2"
    exit 1
fi

echo "=========================================="
echo "T2SQL POC iniciando em: $(hostname)"
echo "Estágio : ${STAGE}"
echo "Smoke   : ${SMOKE:-0}"
echo "Imagem  : ${IMAGE}"
echo "Projeto : ${PROJECT_HOST}"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
echo ""

singularity exec \
    --nv \
    --bind "${RAID}:/workspace" \
    --bind /dev/shm:/dev/shm \
    --pwd "${PROJECT_CTN}" \
    --env HF_HOME="${HF_HOME}" \
    --env OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    --env WANDB_MODE="${WANDB_MODE}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    --env SMOKE="${SMOKE:-0}" \
    --env GEN_PROMPTS="${GEN_PROMPTS:-4000}" \
    --env ITER_TAG="${ITER_TAG:-iter0}" \
    --env PREV_TAG="${PREV_TAG:-iter0scale}" \
    --env CH_EPOCHS="${CH_EPOCHS:-1}" \
    --env SOLVER_THINK_BONUS="${SOLVER_THINK_BONUS:-0.0}" \
    --env DB_ROOT="${DB_ROOT:-}" \
    --env CH_DATA="${CH_DATA:-}" \
    --env VAL_DATA="${VAL_DATA:-}" \
    --env CHALLENGER_REP_LAMBDA="${CHALLENGER_REP_LAMBDA:-0.0}" \
    --env CH_MICRO="${CH_MICRO:-2}" \
    --env SAVE_FREQ="${SAVE_FREQ:-100}" \
    --env BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}" \
    --env SOLVER_DATA="${SOLVER_DATA:-}" \
    --env MICRO_BATCH="${MICRO_BATCH:-8}" \
    --env ROLLOUT_MEM="${ROLLOUT_MEM:-0.4}" \
    --env SOLVER_GPU="${SOLVER_GPU:-}" \
    --env TRAIN_GPUS="${TRAIN_GPUS:-}" \
    --env LOGPROB_MICRO="${LOGPROB_MICRO:-16}" \
    --env NUM_GPUS="${NUM_GPUS:-1}" \
    --env TP="${TP:-1}" \
    "${IMAGE}" \
    bash t2sql_poc_run.sh "${STAGE}"
