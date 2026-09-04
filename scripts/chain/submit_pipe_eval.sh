#!/bin/bash
# Eval no pipeline OmniSQL (protocolo da Tabela 1 do paper) dos 4 checkpoints
# encadeados, lendo os pesos do HF. Roda na gpu-node-a: e o unico no com
# omnisql_pipeline/data e official_eval/. Reusa os scripts ja validados.
#
# Uso: bash submit_pipe_eval.sh [jobid_dependencia ...]
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
DEP=""
if [ $# -gt 0 ]; then DEP="--dependency=afterok:$(IFS=:; echo "$*")"; fi

MODELOS="
anonymous/sql-zero-3b-solver-bird2-think-chained|bird2c
anonymous/sql-zero-3b-solver-bird3-think-chained|bird3c
anonymous/sql-zero-7b-solver-bird2-think-chained|bird2xc
anonymous/sql-zero-7b-solver-bird3-think-chained|bird3xc
"
for linha in $MODELOS; do
  repo=${linha%%|*}; tag=${linha##*|}
  inf=$(sbatch --parsable -J "pipe-infer-$tag" $DEP scripts/nothink_pipe_infer.sbatch "$repo" "$tag")
  sco=$(sbatch --parsable -J "pipe-score-$tag" --dependency=afterok:"$inf" scripts/nothink_pipe_score.sbatch "$tag")
  echo "$tag: infer=$inf score=$sco"
done
