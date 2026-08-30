#!/bin/bash
# OBSOLETO — NÃO SUBMETER. Variante gpu-node-a mantida apenas para registro histórico.
# A cadeia omni20kf roda 100% em gpu-node-b com images/verl0.6-cu128.sif; este script usa a
# imagem gpu-node-a verl0.5-sglang048.sif, e submetê-lo reintroduziria um confundidor de imagem.
# Use os equivalentes scripts/g3_*_omni20kf_*_b200*.sbatch.
#SBATCH --job-name=g3-ch-omni20kf-it2
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
#SBATCH --requeue
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
CKDIR=checkpoints/dr-zero-t2sql/t2sql_solver_bird1_omni20kf_grpo_group8_omnisql-7b
LAST=$(cat $CKDIR/latest_checkpointed_iteration.txt)
SOLVER=./$CKDIR/hf_model_$LAST
[ -f "$SOLVER/config.json" ] || { echo "solver iter1 hf_model ausente: $SOLVER"; exit 1; }
# Derivar os indices das GPUs REALMENTE alocadas pelo Slurm (o script hardcodava 0,1,2,
# que colide com outros jobs quando o Slurm nao me da fisicamente as GPUs 0,1,2).
IFS=',' read -ra _G <<< "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES nao setado pelo Slurm}"
[ "${#_G[@]}" -ge 3 ] || { echo "esperava 3 GPUs, recebi: ${CUDA_VISIBLE_DEVICES}"; exit 1; }
TRAIN_GPUS_V="${_G[0]},${_G[1]}"
SOLVER_GPU_V="${_G[2]}"
echo "Slurm CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} -> TRAIN_GPUS=${TRAIN_GPUS_V} SOLVER_GPU=${SOLVER_GPU_V}"
source scripts/g3_gpu_guard.sh
singularity exec --nv \
  --bind /workspace/anon:/workspace \
  --bind /dev/shm:/dev/shm \
  --pwd /workspace/projects/dr-zero-t2sql \
  --env HF_HOME=/workspace/hf-cache \
  --env PYTHONNOUSERSITE=1 \
  --env WANDB_MODE=online \
  --env WANDB_CACHE_DIR=/workspace/tmp/wandb-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache \
  --env NUM_GPUS=2 --env TP=1 \
  --env CH_MICRO=1 --env LOGPROB_MICRO=8 \
  --env CH_EPOCHS=1 \
  --env CHALLENGER_REP_LAMBDA=0.5 \
  --env BASE_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct \
  --env SOLVER_GPU=${SOLVER_GPU_V} --env TRAIN_GPUS=${TRAIN_GPUS_V} \
  --env DB_ROOT=./bird/all_databases \
  --env CH_DATA=./data/t2sql_zero_bird.parquet \
  --env VAL_DATA=./data/t2sql_test_bird.parquet \
  --env SAVE_FREQ=25 --env MAX_CKPT=1 \
  --env EXEC_TIMEOUT_S=600 \
  /workspace/anon/images/verl0.5-sglang048.sif \
  bash t2sql_challenger.sh bird2_omni20kf "$SOLVER" data.max_prompt_length=8192 actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True actor_rollout_ref.rollout.gpu_memory_utilization=0.20
echo 'G3-CHALLENGER-ITER2-OMNI20KF COMPLETO'
