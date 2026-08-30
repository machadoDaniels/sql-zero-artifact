# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Solver GRPO training with the 3-tier execution reward (Arctic style).
# Reward comparisons go through the SQLite exec server on :8000.
#
# Usage: bash t2sql_solver.sh [train_parquet] [iter_tag] [extra hydra overrides...]

set -x

TRAIN_DATA=${1:-./data/t2sql_solver_iter0.parquet}
iter_tag=${2:-iter0}
if [ $# -ge 1 ]; then shift; fi
if [ $# -ge 1 ]; then shift; fi

tp=${TP:-1}
gpus=${NUM_GPUS:-8}
batch_per_gpu=${MICRO_BATCH:-8}
logprob_per_gpu=${LOGPROB_MICRO:-16}
rollout_memory_utilization=${ROLLOUT_MEM:-0.4}

algorithm=grpo
grpo_group_size=8
model=${SOLVER_INIT:-${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}}
# MODEL_NICK mantém nomes de experimento estáveis quando model é um path local (hf_model)
model_name=${MODEL_NICK:-$(basename "$model" | tr '[:upper:]' '[:lower:]')}

SOLVER_NAME="t2sql_solver_${iter_tag}_${algorithm}_group${grpo_group_size}_${model_name}"
VAL_DATA=${VAL_DATA:-./data/t2sql_test.parquet}

source activate llm 2>/dev/null || true;  # no container (singularity) o env ja e o do sistema
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

export SOLVER_THINK_BONUS=${SOLVER_THINK_BONUS:-0.0}  # >0 liga o braço think (ablação)


# Espera um endpoint de health responder; falha alto em vez de travar p/ sempre.
wait_health() {
    for _ in $(seq 1 360); do
        curl -s "$1" > /dev/null && return 0
        sleep 5
    done
    echo "ERRO: timeout (30min) esperando $1 — servidor não subiu."
    exit 1
}

exec_port=$((18000 + $(( ${SLURM_JOB_ID:-0} % 1000 ))))
export SQL_EXEC_URL="http://127.0.0.1:${exec_port}"
python search/sql_exec_server.py --db_root ${DB_ROOT:-./spider/database} --port ${exec_port} &
EXEC_PID=$!
trap 'kill $EXEC_PID 2>/dev/null || true' EXIT

wait_health ${SQL_EXEC_URL}/health

# save_contents=[model] por padrão: optimizer/extra são scratch (dezenas de GB por step) e
# estourar a quota TRUNCA o checkpoint em escrita. Para treino resumível, passe
# 'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' como argumento
# extra — o $@ vem depois dos defaults, então o override do chamador vence no Hydra.
python -m verl.trainer.main_ppo \
    --config-path="$PWD/config" \
    --config-name='t2sql_grpo' \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=256 \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=${algorithm} \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.actor.grad_clip=0.1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${batch_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=${grpo_group_size} \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${logprob_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${logprob_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=verl/custom_reward/t2sql_reward.py \
    custom_reward_function.name=compute_t2sql_solver_score_batch \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero-t2sql" \
    trainer.experiment_name=$SOLVER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-100} \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-2} \
    trainer.test_freq=25 \
    trainer.val_before_train=False \
    trainer.validation_data_dir="./data/${SOLVER_NAME}" \
    trainer.total_epochs=1 $@
