#!/bin/bash
# Solver da iteracao N ENCADEADO: init de pesos no solver da iteracao N-1
# (SOLVER_INIT), replicando actor_rollout_ref.model.path=<solver_iterN-1_hf>
# do Dr.Zero original. Depois merge + eval oficial greedy (BIRD dev + Spider dev).
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/logs_chain/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
: "${IMG:?}" "${TAG:?}" "${SOLVER_INIT:?}" "${MODEL_NICK:?}" "${TRAIN_PARQUET:?}"

case "$SOLVER_INIT" in
  ./*|/*) [ -f "$SOLVER_INIT/config.json" ] || { echo "SOLVER_INIT ausente (init encadeado): $SOLVER_INIT"; exit 1; };;
  *)      echo "init do Hub (cold-start): $SOLVER_INIT";;
esac
[ -f "$TRAIN_PARQUET" ] || { echo "gen ausente: $TRAIN_PARQUET"; exit 1; }

export SINGULARITYENV_HF_HOME=/workspace/hf-cache
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_WANDB_MODE=online
export SINGULARITYENV_WANDB_CACHE_DIR=/workspace/tmp/wandb-cache
export SINGULARITYENV_XDG_CACHE_HOME=/workspace/tmp/xdg-cache
export SINGULARITYENV_WANDB_DIR=/workspace/tmp/wandb
export SINGULARITYENV_NUM_GPUS=${NUM_GPUS:-2}
export SINGULARITYENV_TP=1
export SINGULARITYENV_MICRO_BATCH=${MICRO_BATCH:-4}
export SINGULARITYENV_LOGPROB_MICRO=8
if [ -n "${ROLLOUT_MEM:-}" ]; then export SINGULARITYENV_ROLLOUT_MEM="$ROLLOUT_MEM"; fi
export SINGULARITYENV_DB_ROOT=./bird/all_databases
export SINGULARITYENV_VAL_DATA=./data/t2sql_test_bird.parquet
export SINGULARITYENV_SAVE_FREQ=25
export SINGULARITYENV_MAX_CKPT=${MAX_CKPT:-1}
export SINGULARITYENV_SOLVER_THINK_BONUS=0.1
SOLVER_INIT_C="${SOLVER_INIT/\/raid\/user_anonymous//workspace}"   # host -> container
export SINGULARITYENV_SOLVER_INIT="$SOLVER_INIT_C"
export SINGULARITYENV_MODEL_NICK="$MODEL_NICK"
export SINGULARITYENV_VAL_DUMP_DIR=/workspace/tmp/valdumps/${TAG}
if [ -n "${SGLANG_ATTENTION_BACKEND:-}" ]; then export SINGULARITYENV_SGLANG_ATTENTION_BACKEND="$SGLANG_ATTENTION_BACKEND"; fi

singularity exec --nv \
  --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  "$IMG" \
  bash t2sql_solver.sh "$TRAIN_PARQUET" "$TAG" data.max_prompt_length=8192 +data.seed=1 trainer.resume_mode=disable "hydra.run.dir=/workspace/tmp/hydra/${TAG}" \
    "trainer.default_local_dir=/workspace/ckpts_chain/t2sql_solver_${TAG}_grpo_group8_${MODEL_NICK}"

# quota: checkpoints/ e setgid sd22 (650G, no limite). ckpts_chain herda o grupo
# pessoal, que tem folga. CKDIR e caminho de HOST; CKC e o mesmo visto do container.
CKDIR="/workspace/anon/ckpts_chain/t2sql_solver_${TAG}_grpo_group8_${MODEL_NICK}"
CKC="${CKDIR/\/raid\/user_anonymous//workspace}"
LAST=$(cat "$CKDIR/latest_checkpointed_iteration.txt")
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 $IMG"
$W python -m verl.model_merger merge --backend fsdp --local_dir "$CKC/global_step_$LAST/actor" --target_dir "$CKC/hf_model_$LAST"
[ -f "$CKDIR/hf_model_$LAST/config.json" ] || { echo "MERGE FALHOU"; exit 1; }
# nome estavel para a proxima iteracao encadear sem conhecer o step final
ln -sfn "hf_model_$LAST" "$CKDIR/hf_model_final"
$W bash t2sql_baseline.sh "$CKC/hf_model_$LAST" --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --out "/workspace/results_chain/${TAG}_s${LAST}_bird.json" --concurrency 48
$W bash t2sql_baseline.sh "$CKC/hf_model_$LAST" --out "/workspace/results_chain/${TAG}_s${LAST}_spider.json" --concurrency 48
echo "SOLVER $TAG COMPLETO (step $LAST)"
