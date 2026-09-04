#!/bin/bash
# Sobe para o HF os dois checkpoints de UMA iteracao encadeada: o challenger
# (hf_model, mergeado no passo de gen) e o solver (hf_model_final). Repos
# privados. CPU-only, fora do caminho critico da cadeia.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/logs_chain/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
: "${IMG:?}" "${SOLVER_DIR:?}" "${SOLVER_REPO:?}"
CH_DIR=${CH_DIR:-}; CH_REPO=${CH_REPO:-}

export SINGULARITYENV_HF_HOME=/workspace/hf-cache
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_XDG_CACHE_HOME=/workspace/tmp/xdg-cache
# host -> container: /workspace/anon esta montado como /workspace
export SINGULARITYENV_CH_DIR="${CH_DIR/\/raid\/user_anonymous//workspace}"
export SINGULARITYENV_CH_REPO="$CH_REPO"
export SINGULARITYENV_SOLVER_DIR="${SOLVER_DIR/\/raid\/user_anonymous//workspace}"
export SINGULARITYENV_SOLVER_REPO="$SOLVER_REPO"

singularity exec --bind /workspace/anon:/workspace \
  --pwd /workspace/projects/dr-zero-t2sql "$IMG" python - <<'PYEOF'
import os, glob
from huggingface_hub import HfApi
api = HfApi()
jobs = [(os.environ["SOLVER_DIR"], os.environ["SOLVER_REPO"])]
if os.environ.get("CH_DIR"):
    jobs.insert(0, (os.environ["CH_DIR"], os.environ["CH_REPO"]))
for path, repo in jobs:
    path = os.path.realpath(path)  # solver aponta p/ symlink hf_model_final
    assert os.path.isfile(os.path.join(path, "config.json")), f"faltou config.json em {path}"
    shards = glob.glob(os.path.join(path, "*.safetensors"))
    assert shards, f"sem shards em {path}"  # armadilha 6: dir existir != checkpoint existir
    api.create_repo(repo, exist_ok=True, private=True)
    print(f"subindo {path} ({len(shards)} shards) -> {repo}", flush=True)
    api.upload_folder(folder_path=path, repo_id=repo,
                      commit_message=f"upload {os.path.basename(path)} (espiral encadeada)")
    info = api.repo_info(repo, files_metadata=True)
    remote = {f.rfilename for f in info.siblings}
    local = {f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}
    missing = sorted(local - remote)
    assert not missing, f"upload incompleto em {repo}: {missing}"
    print(f"OK {repo} ({len(local)} arquivos verificados)", flush=True)
print("UPLOADS COMPLETOS")
PYEOF
