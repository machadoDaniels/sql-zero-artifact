#!/bin/bash
# Gera o dataset sintetico da iteracao N a partir do challenger encadeado.
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/workspace/anon/logs_chain/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
: "${IMG:?}" "${CHDIR:?}" "${OUT_PARQUET:?}"

LAST=$(cat "$CHDIR/latest_checkpointed_iteration.txt")
CHC="${CHDIR/\/raid\/user_anonymous//workspace}"   # mesmo dir visto de dentro do container
# HF_OUT separa a LEITURA dos shards do DESTINO do merge: o ch2 do 3B ficou no
# checkpoints/ antigo (grupo sd22, quota estourada) e nao aceita escrita.
HF_OUT=${HF_OUT:-$CHDIR/hf_model}
HFC="${HF_OUT/\/raid\/user_anonymous//workspace}"
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env PYTHONNOUSERSITE=1 --env HF_HOME=/workspace/hf-cache --env SLURM_JOB_ID=$SLURM_JOB_ID $IMG"
if [ ! -f "$HF_OUT/config.json" ]; then
  $W python -m verl.model_merger merge --backend fsdp --local_dir "$CHC/global_step_$LAST/actor" --target_dir "$HFC"
fi
[ -f "$HF_OUT/config.json" ] || { echo "MERGE FALHOU"; exit 1; }

$W bash -c "
PORT=\$((24000 + ${SLURM_JOB_ID:-0} % 1000))
python -m sglang.launch_server --model=$HFC --port=\$PORT --mem-fraction-static=0.85 --log-level=error &
SP=\$!
trap \"kill \$SP 2>/dev/null || true\" EXIT
for _ in \$(seq 1 240); do curl -s http://127.0.0.1:\$PORT/health >/dev/null && break; sleep 5; done
curl -s http://127.0.0.1:\$PORT/health >/dev/null || { echo ERRO sglang; exit 1; }
python t2sql_data.py --sample_rows 2 gen-solver-data --endpoint http://127.0.0.1:\$PORT/v1 --model ./$CHDIR/hf_model --db_root ./bird/train_databases --db_list ./bird/train_dbs_keep.txt --num_prompts 12000 --samples_per_prompt 5 --max_pairs 20000 --out $OUT_PARQUET --concurrency 64
"
[ -f "$OUT_PARQUET" ] || { echo "GEN FALHOU: $OUT_PARQUET"; exit 1; }
echo "GEN $OUT_PARQUET COMPLETO"
