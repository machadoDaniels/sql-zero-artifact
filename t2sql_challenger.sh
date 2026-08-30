# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Challenger (proposer) GRPO training. The reward needs two local services:
#   :8000 SQLite exec server  -> gold validation + result-set comparison
#   :8001 sglang solver server -> n single-shot solver rollouts (difficulty)
#
# At iter1 the solver server runs the base model; at iter2+ pass the
# converted (HF) solver checkpoint from the previous iteration.
#
# grpo_batch groups the advantage baseline by data_source, which carries the
# sampled complexity level (t2sql_zero_<level>) -> HRPO-lite by complexity.
#
# Usage: bash t2sql_challenger.sh [iter_tag] [solver_model] [extra hydra overrides...]

set -x

iter_tag=${1:-iter1}
solver_model=${2:-Qwen/Qwen2.5-Coder-3B-Instruct}
if [ $# -ge 1 ]; then shift; fi
if [ $# -ge 1 ]; then shift; fi

tp=${TP:-1}
dp=${SOLVER_DP:-1}
gpus=${NUM_GPUS:-8}
batch_per_gpu=${CH_MICRO:-2}
logprob_per_gpu=${LOGPROB_MICRO:-16}
rollout_memory_utilization=0.25

algorithm=grpo_batch
grpo_group_size=1
reward_group_size=5
model=${CH_INIT:-${BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}}
# CH_INIT encadeia o challenger na iteracao anterior (Dr.Zero: trainer.resume_from_path).
# Aqui init de pesos e equivalente: KL desligado (use_kl_loss=False), logo a ref nunca e usada.
# MODEL_NICK mantem nomes de experimento estaveis quando model e um path local (hf_model)
model_name=${MODEL_NICK:-$(basename "$model" | tr '[:upper:]' '[:lower:]')}

CHALLENGER_NAME="t2sql_challenger_${iter_tag}_${algorithm}_group${grpo_group_size}-${reward_group_size}_${model_name}"

TRAIN_DATA=${CH_DATA:-./data/t2sql_zero.parquet}
VAL_DATA=${VAL_DATA:-./data/t2sql_test.parquet}

source activate llm 2>/dev/null || true;  # no container (singularity) o env ja e o do sistema
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true

export CHALLENGER_REP_LAMBDA=${CHALLENGER_REP_LAMBDA:-0.0}  # >0 liga penalidade de repeticao (R-Zero)


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
solver_port=$((19000 + $(( ${SLURM_JOB_ID:-0} % 1000 ))))
export SQL_EXEC_URL="http://127.0.0.1:${exec_port}"
python search/sql_exec_server.py --db_root ${DB_ROOT:-./spider/database} --port ${exec_port} &
EXEC_PID=$!

# SOLVER_GPU: GPU dedicada ao servidor do solver congelado (necessário no 7B,
# onde servidor + engine de rollout + FSDP não cabem juntos em 80GB).
# Vazio (default) = compartilha as GPUs do treino, como sempre foi no 3B.
if [ -n "${SOLVER_GPU:-}" ]; then
    CUDA_VISIBLE_DEVICES=${SOLVER_GPU} python -m sglang.launch_server \
        --model=${solver_model} \
        --port=${solver_port} \
        --mem-fraction-static=0.85 \
        --dp-size=1 \
        --tp-size=1 \
        --log-level=error &
else
    python -m sglang.launch_server \
        --model=${solver_model} \
        --port=${solver_port} \
        --mem-fraction-static=${rollout_memory_utilization} \
        --dp-size=${dp} \
        --tp-size=${tp} \
        --log-level=error &
fi
SOLVER_PID=$!
trap 'kill $EXEC_PID $SOLVER_PID 2>/dev/null || true' EXIT

wait_health ${SQL_EXEC_URL}/health
wait_health http://127.0.0.1:${solver_port}/health

# Com servidor em GPU dedicada, o treino fica restrito às demais.
if [ -n "${SOLVER_GPU:-}" ]; then
    export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-0,1}
fi

# save_contents=[model] por padrão: optimizer/extra são scratch (dezenas de GB por step) e
# estourar a quota TRUNCA o checkpoint em escrita. Para treino resumível, passe
# 'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]' como argumento
# extra — o $@ vem depois dos defaults, então o override do chamador vence no Hydra.
python -m verl.trainer.main_ppo \
    --config-path="$PWD/config" \
    --config-name='t2sql_grpo' \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
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
    custom_reward_function.name=compute_t2sql_challenger_score_batch \
    custom_reward_function.reward_kwargs.model_name=${solver_model} \
    custom_reward_function.reward_kwargs.base_url="http://127.0.0.1:${solver_port}" \
    custom_reward_function.reward_kwargs.reward_rollout_n=${reward_group_size} \
    trainer.logger='["wandb", "console"]' \
    trainer.project_name="dr-zero-t2sql" \
    trainer.experiment_name=$CHALLENGER_NAME \
    trainer.n_gpus_per_node=${gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-100} \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model]' \
    trainer.max_actor_ckpt_to_keep=${MAX_CKPT:-2} \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=${CH_EPOCHS:-1} $@
