#!/bin/bash
# Baixa os checkpoints da iteracao 1 (challenger + solver) que servem de init
# encadeado para a iteracao 2. Alguns so existem no Hub (g1b-s78, bird1_7b),
# por isso este passo. CPU-only: nao pede GPU.
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/logs_chain/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
: "${IMG:?}" "${CH1_REPO:?}" "${CH1_DIR:?}" "${SOLVER1_REPO:?}" "${SOLVER1_DIR:?}"

export SINGULARITYENV_HF_HOME=/workspace/hf-cache
export SINGULARITYENV_PYTHONNOUSERSITE=1
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql $IMG"

fetch() {  # repo, destino
  if [ -f "$2/config.json" ]; then echo "ja existe: $2"; return 0; fi
  $W python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$1', local_dir='$2')
print('baixado em', p)
"
  [ -f "$2/config.json" ] || { echo "DOWNLOAD FALHOU: $1"; exit 1; }
}

mkdir -p "$(dirname "$CH1_DIR")"
fetch "$CH1_REPO" "$CH1_DIR"
fetch "$SOLVER1_REPO" "$SOLVER1_DIR"

# checagem dura: shards presentes, nao so o config (armadilha 6 do CLAUDE.md)
for d in "$CH1_DIR" "$SOLVER1_DIR"; do
  n=$(ls "$d"/*.safetensors 2>/dev/null | wc -l)
  [ "$n" -ge 1 ] || { echo "SEM SHARDS em $d"; exit 1; }
  echo "$d: $n shards, $(du -sh "$d" | cut -f1)"
done
echo "FETCH OK"
