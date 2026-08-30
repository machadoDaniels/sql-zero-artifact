# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Estágios da POC — roda DENTRO do container (chamado por t2sql_poc.sbatch).
# Não submeta este arquivo direto; use: sbatch t2sql_poc.sbatch [stage]
#
# Estágios (idempotente: pula o que já tem output):
#   prep     -> valida Spider + gera o dev set (t2sql_test.parquet)
#   baseline -> passo 0: zero-shot do modelo base no Spider dev
#   gen0     -> iter0: dados sintéticos com o modelo base como proposer
#   solver0  -> iter0: treino GRPO do solver + eval no dev

set -euo pipefail

# O Slurm do cluster exporta ROCR/HIP_VISIBLE_DEVICES (AMD) junto com
# CUDA_VISIBLE_DEVICES; o verl recusa a combinacao em GPUs NVIDIA.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

# Persistir logs do Ray fora do overlay do container (debug de workers mortos)
export RAY_TMPDIR=${RAY_TMPDIR:-$PWD/logs/ray_tmp}
mkdir -p "$RAY_TMPDIR"

STAGE=${1:-all}
MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
MODEL_NAME=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
SOLVER_DATA="${SOLVER_DATA:-./data/t2sql_solver_iter0.parquet}"

# Smoke test (docs/DECISOES_ARQUITETURA.md §8): integração com custo mínimo.
if [ "${SMOKE:-0}" = "1" ]; then
    EVAL_ARGS="--limit 50"
    GEN_PROMPTS=50
    SOLVER_ARGS="data.train_batch_size=16 actor_rollout_ref.actor.ppo_mini_batch_size=16 trainer.save_freq=-1 trainer.test_freq=5"
else
    EVAL_ARGS=""
    GEN_PROMPTS=${GEN_PROMPTS:-4000}
    SOLVER_ARGS=""
fi

mkdir -p logs results data

run_stage() {
    case "$STAGE" in
        all)        case "$1" in prep|baseline|gen0|solver0|eval0) true;; *) false;; esac ;;
        iter|iter1) case "$1" in challenger1|gen1|solver1) true;; *) false;; esac ;;
        *)          [ "$STAGE" = "$1" ] ;;
    esac
}

TAG="${ITER_TAG:-iter0}"
PREV_TAG="${PREV_TAG:-iter0scale}"
SOLVER_HF="./checkpoints/dr-zero-t2sql/t2sql_solver_${PREV_TAG}_grpo_group8_${MODEL_NAME}/hf_model"

# ---------------------------------------------------------------- prep
if run_stage prep; then
    if [ ! -d ./spider/database ]; then
        echo "ERRO: ./spider/database não existe. Rode antes:"
        echo "  python scripts/download_spider.py --zip_path <spider_data.zip>"
        exit 1
    fi
    if [ ! -f ./data/t2sql_test.parquet ]; then
        python t2sql_data.py --spider_dir ./spider test-set --out ./data/t2sql_test.parquet
    fi
fi

# ------------------------------------------------------------ baseline
if run_stage baseline; then
    if [ ! -f "./results/baseline_${MODEL_NAME}.json" ]; then
        bash t2sql_baseline.sh "$MODEL" $EVAL_ARGS
    else
        echo "baseline já existe: results/baseline_${MODEL_NAME}.json (pulando)"
    fi
fi

# ---------------------------------------------------------------- gen0
if run_stage gen0; then
    if [ ! -f "$SOLVER_DATA" ]; then
        bash t2sql_gen_data.sh "$MODEL" "$SOLVER_DATA" "$GEN_PROMPTS"
    else
        echo "dados iter0 já existem: $SOLVER_DATA (pulando)"
    fi
fi

# ------------------------------------------------------------- solver0
if run_stage solver0; then
    bash t2sql_solver.sh "$SOLVER_DATA" "${ITER_TAG:-iter0}" $SOLVER_ARGS
fi

# --------------------------------------------------------------- eval0
# Converte o último ckpt FSDP do solver p/ HF e avalia em Spider e BIRD.
if run_stage eval0; then
    if [ -f "./results/${TAG}_spider.json" ] && [ -f "./results/${TAG}_bird.json" ]; then
        echo "eval0 já feito p/ ${TAG} (pulando)"
    else
    CKPT_ROOT="./checkpoints/dr-zero-t2sql/t2sql_solver_${TAG}_grpo_group8_${MODEL_NAME}"
    HF_DIR="${CKPT_ROOT}/hf_model"
    if [ ! -f "$HF_DIR/config.json" ]; then
        last_step=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed "s/.*global_step_//" | sort -n | tail -1)
        if [ -z "$last_step" ]; then
            echo "ERRO: nenhum checkpoint em $CKPT_ROOT"; exit 1
        fi
        python -m verl.model_merger merge --backend fsdp \
            --local_dir "$CKPT_ROOT/global_step_${last_step}/actor" \
            --target_dir "$HF_DIR"
    fi
    bash t2sql_baseline.sh "$HF_DIR" --out "./results/${TAG}_spider.json" $EVAL_ARGS
    bash t2sql_baseline.sh "$HF_DIR" \
        --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases \
        --exec_timeout 30 --out "./results/${TAG}_bird.json" $EVAL_ARGS
    fi
fi

# =================================================================== iter1
# Espiral completa (docs §6): challenger treina contra o solver anterior,
# gen1 usa o challenger treinado, solver1 continua do solver anterior.

CH_ROOT="./checkpoints/dr-zero-t2sql/t2sql_challenger_${TAG}_grpo_batch_group1-5_${MODEL_NAME}"

# --------------------------------------------------------- challenger1
if run_stage challenger1; then
    if ls "$CH_ROOT"/global_step_* > /dev/null 2>&1; then
        echo "challenger1: checkpoint já existe em $CH_ROOT (pulando)"
    else
        [ -d "$SOLVER_HF" ] || { echo "ERRO: solver HF ausente: $SOLVER_HF (rode eval0 de ${PREV_TAG})"; exit 1; }
        [ -f ./data/t2sql_zero.parquet ] || { echo "ERRO: rode t2sql_data.py challenger-prompts antes"; exit 1; }
        bash t2sql_challenger.sh "$TAG" "$SOLVER_HF"
    fi
fi

# ---------------------------------------------------------------- gen1
if run_stage gen1; then
    CH_HF="${CH_ROOT}/hf_model"
    if [ ! -f "$CH_HF/config.json" ]; then
        last_step=$(ls -d "$CH_ROOT"/global_step_* 2>/dev/null | sed "s/.*global_step_//" | sort -n | tail -1)
        [ -z "$last_step" ] && { echo "ERRO: nenhum ckpt de challenger em $CH_ROOT"; exit 1; }
        python -m verl.model_merger merge --backend fsdp \
            --local_dir "$CH_ROOT/global_step_${last_step}/actor" --target_dir "$CH_HF"
    fi
    if [ ! -f "./data/t2sql_solver_${TAG}.parquet" ]; then
        bash t2sql_gen_data.sh "$CH_HF" "./data/t2sql_solver_${TAG}.parquet" "$GEN_PROMPTS"
    fi
fi

# -------------------------------------------------------------- solver1
if run_stage solver1; then
    [ -d "$SOLVER_HF" ] || { echo "ERRO: solver HF ausente: $SOLVER_HF"; exit 1; }
    MODEL_NICK="qwen2.5-coder-3b-instruct" SOLVER_INIT="$SOLVER_HF" \
        bash t2sql_solver.sh "./data/t2sql_solver_${TAG}.parquet" "$TAG" $SOLVER_ARGS
fi

echo "POC ($STAGE) concluída."
