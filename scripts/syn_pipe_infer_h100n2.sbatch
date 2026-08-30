#!/bin/bash
#SBATCH --job-name=syn-pipe-infer-n2
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql/omnisql_pipeline
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
IMG=/workspace/anon/images/vllm_openai_latest.sif
G3LOCAL=/workspace/projects/dr-zero-t2sql/checkpoints/dr-zero-t2sql/t2sql_solver_bird1_omni20k_grpo_group8_omnisql-7b/hf_model_53
run() {
  local MODEL=$1 TAG=$2
  singularity exec --nv --bind /workspace/anon:/workspace     --pwd /workspace/projects/dr-zero-t2sql/omnisql_pipeline     --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1     $IMG python3 infer.py       --pretrained_model_name_or_path $MODEL       --input_file ../official_eval/work/syn_pipe_prompts.json       --output_file ../official_eval/work/${TAG}_syn_pred.json       --tensor_parallel_size 1 --n 1 --temperature 0.0
  echo "SYN_INFER_DONE $TAG"
}
run seeklhy/OmniSQL-7B omnisql7b
run $G3LOCAL g3it1
echo SYN_PIPE_INFER_COMPLETO
