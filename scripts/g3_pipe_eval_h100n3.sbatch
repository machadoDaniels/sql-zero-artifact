#!/bin/bash
#SBATCH --job-name=g3-pipe-omni
#SBATCH --partition=gpu-node-a
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -uo pipefail
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
ROOT=/workspace/anon
HERE=$ROOT/projects/dr-zero-t2sql/omnisql_pipeline
cd $HERE
MODEL=anonymous/sql-zero-7b-solver-omnisql-bird
TAG=g3_iter1_7b
mkdir -p results

run_infer () {  # $1=input  $2=output
  singularity exec --nv \
    --bind $ROOT:/workspace \
    --pwd /workspace/projects/dr-zero-t2sql/omnisql_pipeline \
    --env HF_HOME=/workspace/hf-cache \
    --env HF_TOKEN="$(cat $ROOT/hf-cache/token)" \
    --env PYTHONNOUSERSITE=1 \
    $ROOT/images/vllm_openai_latest.sif \
    python3 infer.py --pretrained_model_name_or_path $MODEL \
      --input_file "$1" --output_file "$2" \
      --tensor_parallel_size 1 --n 1 --temperature 0.0
}

echo "=== BIRD dev ==="
run_infer ./data/dev_bird.json ./results/${TAG}_bird_greedy.json
python3 evaluate_bird.py --pred $HERE/results/${TAG}_bird_greedy.json \
  --gold ../bird/dev_20240627/dev.json --db_path ../bird/dev_databases \
  --mode greedy_search 2>&1 | tail -6

echo "=== Spider test (inferencia; score depois) ==="
run_infer ./data/test_spider.json ./results/${TAG}_spidertest_greedy.json

echo "G3-PIPE-EVAL COMPLETO"
