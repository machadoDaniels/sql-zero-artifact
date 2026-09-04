#!/bin/bash
# Submete a cadeia ENCADEADA (iter2 e iter3) de um braco, com dependencias afterok.
# Cada iteracao: challenger init no challenger anterior -> gen -> solver init no
# solver anterior -> merge + eval oficial. Corrige o desvio em que toda iteracao
# reiniciava do modelo base (o Dr.Zero original encadeia os pesos dos dois papeis).
#
# Uso: bash submit_chain.sh 3b|7b [estagio_inicial] [jobid_dependencia]
set -euo pipefail
ARM=${1:?uso: submit_chain.sh 3b|7b [estagio_inicial] [jobid_dependencia]}
START=${2:-fetch}   # fetch|ch2|gen2|solver2|ch3|gen3|solver3 — retoma a cadeia deste ponto
DEP0=${3:-}         # job do qual o primeiro estagio submetido depende (afterok)
ROOT=/workspace/anon/projects/dr-zero-t2sql
S=$ROOT/scripts/chain
CK=/workspace/anon/ckpts_chain
INIT=./checkpoints/chain_init

# Knobs identicos nos dois bracos: a unica diferenca permitida e o tamanho do modelo
# (e o par no/imagem que o hardware impoe: B200/sm_100 exige verl0.6 + flashinfer).
SOLVER_KNOBS_COMUNS="MICRO_BATCH=2,ROLLOUT_MEM=0.2,MAX_CKPT=1"
CH_DB_ROOT_COMUM=./bird/all_databases
PART=gpu-node-b; GRES=gpu:b200:2; GEN_GRES=gpu:b200:1
IMG=/workspace/anon/images/verl0.6-cu128.sif
SGENV_COMUM="SGLANG_ATTENTION_BACKEND=flashinfer"

case "$ARM" in
  3b)
    NICK=qwen2.5-coder-3b-instruct
    CH1_REPO=anonymous/sql-zero-3b-challenger-bird1
    SOLVER1_REPO=anonymous/sql-zero-3b-solver-bird1b-think
    CH_DB_ROOT="$CH_DB_ROOT_COMUM"
    SOLVER_KNOBS="$SOLVER_KNOBS_COMUNS"
    SGENV="$SGENV_COMUM"
    T2C=bird2c_3b;  T2S=bird2c20k;  P2=./data/t2sql_solver_bird2c_20k.parquet
    T3C=bird3c_3b;  T3S=bird3c20k;  P3=./data/t2sql_solver_bird3c_20k.parquet
    HF=anonymous/sql-zero-3b
    ;;
  7b)
    NICK=qwen2.5-coder-7b-instruct
    CH1_REPO=anonymous/sql-zero-7b-challenger-bird1
    SOLVER1_REPO=anonymous/sql-zero-7b-solver-bird1-think
    CH_DB_ROOT="$CH_DB_ROOT_COMUM"
    SOLVER_KNOBS="$SOLVER_KNOBS_COMUNS"
    SGENV="$SGENV_COMUM"
    T2C=bird2xc_7b; T2S=bird2xc20k; P2=./data/t2sql_solver_bird2xc_7b_20k.parquet
    T3C=bird3xc_7b; T3S=bird3xc20k; P3=./data/t2sql_solver_bird3xc_7b_20k.parquet
    HF=anonymous/sql-zero-7b
    ;;
  *) echo "braco invalido: $ARM"; exit 1;;
esac

CH1=$INIT/ch1_$ARM
SOLVER1=$INIT/solver1_$ARM
chdir_of() { echo "$CK/t2sql_challenger_${1}_grpo_batch_group1-5_${NICK}"; }
sdir_of()  { echo "$CK/t2sql_solver_${1}_grpo_group8_${NICK}"; }

sub() {  # nome, dependencia (ou ""), gres, script, env
  local name=$1 dep=$2 gres=$3 script=$4 env=$5
  local args=(--parsable -J "$name" -p "$PART" --export="ALL,IMG=$IMG,$env")
  if [ "$gres" != none ]; then args+=(--gres="$gres"); fi
  if [ -n "$dep" ]; then args+=(--dependency=afterok:"$dep"); fi
  sbatch "${args[@]}" "$script"
}

E="MODEL_NICK=$NICK"; if [ -n "$SGENV" ]; then E="$E,$SGENV"; fi

run=0
prev="$DEP0"
stage() {  # estagio, gres, script, env
  local st=$1 gres=$2 script=$3 env=$4
  if [ "$run" = 0 ]; then
    if [ "$st" = "$START" ]; then run=1; else return 0; fi
  fi
  prev=$(sub "chain-$ARM-$st" "$prev" "$gres" "$script" "$env")
  printf '%s=%s ' "$st" "$prev"
}

# upload do HF pendura no estagio corrente sem entrar no caminho critico:
# nao atualiza $prev, entao a iteracao seguinte nao espera o upload.
branch() {  # sufixo, env
  local sufixo=$1 env=$2 id
  [ "$run" = 1 ] || return 0
  id=$(sub "chain-$ARM-$sufixo" "$prev" none "$S/chain_hf_upload.sbatch" "$env")
  printf '%s=%s ' "$sufixo" "$id"
}

printf 'braco %s em %s: ' "$ARM" "$PART"
stage fetch   none       "$S/chain_fetch.sbatch" \
  "CH1_REPO=$CH1_REPO,CH1_DIR=$CH1,SOLVER1_REPO=$SOLVER1_REPO,SOLVER1_DIR=$SOLVER1"
stage ch2     "$GRES"    "$S/chain_challenger.sbatch" \
  "$E,TAG=$T2C,CH_INIT=$CH1,SOLVER=$SOLVER1,CH_DB_ROOT=$CH_DB_ROOT"
stage gen2    "$GEN_GRES" "$S/chain_gen.sbatch" \
  "$E,CHDIR=$(chdir_of "$T2C"),OUT_PARQUET=$P2"
stage solver2 "$GRES"    "$S/chain_solver_eval.sbatch" \
  "$E,$SOLVER_KNOBS,TAG=$T2S,SOLVER_INIT=$SOLVER1,TRAIN_PARQUET=$P2"
branch up2 "IMG=$IMG,CH_DIR=$(chdir_of "$T2C")/hf_model,CH_REPO=$HF-challenger-bird2-chained,SOLVER_DIR=$(sdir_of "$T2S")/hf_model_final,SOLVER_REPO=$HF-solver-bird2-think-chained"
stage ch3     "$GRES"    "$S/chain_challenger.sbatch" \
  "$E,TAG=$T3C,CH_INIT=$(chdir_of "$T2C")/hf_model,SOLVER=$(sdir_of "$T2S")/hf_model_final,CH_DB_ROOT=$CH_DB_ROOT"
stage gen3    "$GEN_GRES" "$S/chain_gen.sbatch" \
  "$E,CHDIR=$(chdir_of "$T3C"),OUT_PARQUET=$P3"
stage solver3 "$GRES"    "$S/chain_solver_eval.sbatch" \
  "$E,$SOLVER_KNOBS,TAG=$T3S,SOLVER_INIT=$(sdir_of "$T2S")/hf_model_final,TRAIN_PARQUET=$P3"
branch up3 "IMG=$IMG,CH_DIR=$(chdir_of "$T3C")/hf_model,CH_REPO=$HF-challenger-bird3-chained,SOLVER_DIR=$(sdir_of "$T3S")/hf_model_final,SOLVER_REPO=$HF-solver-bird3-think-chained"
echo
