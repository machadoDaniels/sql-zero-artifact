#!/bin/bash
# Challenger da iteracao N ENCADEADO: init de pesos no challenger da iteracao N-1
# (CH_INIT), replicando trainer.resume_from_path do Dr.Zero original. O solver
# da iteracao N-1 entra como referee congelado, como nos runs cold-start.
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/logs_chain/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
: "${IMG:?}" "${TAG:?}" "${CH_INIT:?}" "${SOLVER:?}" "${MODEL_NICK:?}" "${CH_DB_ROOT:?}"

[ -f "$CH_INIT/config.json" ] || { echo "CH_INIT ausente (init encadeado): $CH_INIT"; exit 1; }
[ -f "$SOLVER/config.json" ] || { echo "SOLVER referee ausente: $SOLVER"; exit 1; }

export SINGULARITYENV_HF_HOME=/workspace/hf-cache
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_WANDB_MODE=online
export SINGULARITYENV_WANDB_CACHE_DIR=/workspace/tmp/wandb-cache
export SINGULARITYENV_XDG_CACHE_HOME=/workspace/tmp/xdg-cache
export SINGULARITYENV_WANDB_DIR=/workspace/tmp/wandb
export SINGULARITYENV_NUM_GPUS=${NUM_GPUS:-2}
export SINGULARITYENV_TP=1
export SINGULARITYENV_CH_MICRO=1
export SINGULARITYENV_LOGPROB_MICRO=8
export SINGULARITYENV_CH_EPOCHS=1
export SINGULARITYENV_CHALLENGER_REP_LAMBDA=0.5
export SINGULARITYENV_DB_ROOT="$CH_DB_ROOT"
export SINGULARITYENV_CH_DATA=./data/t2sql_zero_bird.parquet
export SINGULARITYENV_VAL_DATA=./data/t2sql_test_bird.parquet
export SINGULARITYENV_SAVE_FREQ=25
export SINGULARITYENV_MAX_CKPT=${MAX_CKPT:-1}   # 3 encheu o /raid da b200 (exit 120)
CH_INIT_C="${CH_INIT/\/raid\/user_anonymous//workspace}"   # host -> container
SOLVER_C="${SOLVER/\/raid\/user_anonymous//workspace}"
export SINGULARITYENV_CH_INIT="$CH_INIT_C"
export SINGULARITYENV_MODEL_NICK="$MODEL_NICK"
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then export SINGULARITYENV_SGLANG_ATTENTION_BACKEND="$SGLANG_ATTENTION_BACKEND"; fi

singularity exec --nv \
  --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  "$IMG" \
  bash t2sql_challenger.sh "$TAG" "$SOLVER_C" data.max_prompt_length=8192 +data.seed=1 trainer.resume_mode=disable "hydra.run.dir=/workspace/tmp/hydra/${TAG}" \
    "trainer.default_local_dir=/workspace/ckpts_chain/t2sql_challenger_${TAG}_grpo_batch_group1-5_${MODEL_NICK}"
echo "CHALLENGER $TAG COMPLETO"
