#!/bin/bash
#SBATCH --job-name=score-st-gate
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql/official_eval/test-suite-sql-eval
singularity exec --bind /workspace/anon:/workspace   --pwd /workspace/projects/dr-zero-t2sql/official_eval/test-suite-sql-eval   --env PYTHONNOUSERSITE=1 --env PYTHONPATH=/workspace/python-packages/eval-extra --env NLTK_DATA=/workspace/nltk_data /workspace/anon/images/verl0.5-sglang048.sif   python3 evaluation.py     --gold ../work/gold_spidertest.txt     --pred ../work/pred_g1b3b_spidertest.txt     --db ../../spider/test_database     --table ../../spider/test_tables.json     --etype exec --plug_value --keep_distinct --progress_bar_for_each_datapoint
echo SCORE_DONE
