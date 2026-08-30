#!/bin/bash
#SBATCH --job-name=g3-pipe-infer
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
ROOT=/workspace/anon
HERE=$ROOT/projects/dr-zero-t2sql/omnisql_pipeline
cd "$HERE"
MODEL="$1"   # local path relative to omnisql_pipeline, e.g. ./ckpts/g3_iter2_7b
TAG="$2"     # e.g. g3_iter2_7b
IMG=$ROOT/images/vllm_openai_latest.sif
mkdir -p results

run_infer () {  # $1=model $2=input $3=output
  singularity exec --nv \
    --bind "$ROOT":/workspace \
    --pwd /workspace/projects/dr-zero-t2sql/omnisql_pipeline \
    --env HF_HOME=/workspace/hf-cache \
    --env PYTHONNOUSERSITE=1 \
    "$IMG" \
    python3 infer.py --pretrained_model_name_or_path "$1" \
      --input_file "$2" --output_file "$3" \
      --tensor_parallel_size 1 --n 1 --temperature 0.0
}

echo "=== BIRD dev ($TAG) ==="
run_infer "$MODEL" ./data/dev_bird.json ./results/${TAG}_bird_greedy.json
python3 evaluate_bird.py --pred "$HERE/results/${TAG}_bird_greedy.json" \
  --gold ../bird/dev_20240627/dev.json --db_path ../bird/dev_databases \
  --mode greedy_search 2>&1 | tail -6

echo "=== Spider test ($TAG) ==="
run_infer "$MODEL" ./data/test_spider.json ./results/${TAG}_spidertest_greedy.json

echo "=== Spider-Syn ($TAG) ==="
run_infer "$MODEL" ../official_eval/work/syn_pipe_prompts.json ../official_eval/work/${TAG}_syn_pred.json

echo "G3-PIPE-INFER-COMPLETO $TAG"
