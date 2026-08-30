#!/bin/bash
#SBATCH --job-name=bird3-gen-7b
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env SLURM_JOB_ID=$SLURM_JOB_ID /workspace/anon/images/verl0.6-cu128.sif"
CHDIR=checkpoints/dr-zero-t2sql/t2sql_challenger_bird3_7b_grpo_batch_group1-5_qwen2.5-coder-7b-instruct
LAST=$(cat $CHDIR/latest_checkpointed_iteration.txt)
if [ ! -f "$CHDIR/hf_model/config.json" ]; then
  $W python -m verl.model_merger merge --backend fsdp --local_dir ./$CHDIR/global_step_$LAST/actor --target_dir ./$CHDIR/hf_model
fi
[ -f "$CHDIR/hf_model/config.json" ] || { echo "MERGE FALHOU"; exit 1; }
$W bash -c "
PORT=\$((24000 + ${SLURM_JOB_ID:-0} % 1000))
python -m sglang.launch_server --model=./$CHDIR/hf_model --port=\$PORT --mem-fraction-static=0.85 --log-level=error &
SP=\$!
trap \"kill \$SP 2>/dev/null || true\" EXIT
for _ in \$(seq 1 240); do curl -s http://127.0.0.1:\$PORT/health >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:\$PORT/health >/dev/null || { echo ERRO sglang; exit 1; }
python t2sql_data.py --sample_rows 2 gen-solver-data --endpoint http://127.0.0.1:\$PORT/v1 --model ./$CHDIR/hf_model --db_root ./bird/train_databases --db_list ./bird/train_dbs_keep.txt --num_prompts 12000 --samples_per_prompt 5 --max_pairs 20000 --out ./data/t2sql_solver_bird3_7b_20k.parquet --concurrency 64
"
