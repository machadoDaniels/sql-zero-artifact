#!/bin/bash
#SBATCH --job-name=hf-availability-audit
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.6-cu128.sif"
$W python - << "PYEOF"
from huggingface_hub import HfApi

api = HfApi()

BIRD1 = "anonymous/sql-zero-7b-solver-bird1-think"

def describe(repo, tag=""):
    try:
        info = api.repo_info(repo, files_metadata=True)
    except Exception as e:
        print(f"REPO {repo} {tag} :: ERRO {type(e).__name__}")
        return None
    files = {s.rfilename: s.size for s in info.siblings}
    cfg = files.get("config.json")
    shards = {k: v for k, v in files.items() if k.endswith(".safetensors")}
    print(f"REPO {repo} {tag}")
    print(f"  private={info.private} usedStorage={getattr(info,'usedStorage',None)} sha={info.sha} lastmod={info.lastModified}")
    print(f"  config.json_size={cfg}")
    print(f"  n_shards={len(shards)} total_shard_bytes={sum(v or 0 for v in shards.values())}")
    for k in sorted(shards):
        print(f"    {k} = {shards[k]}")
    try:
        refs = api.list_repo_refs(repo)
        print("  branches:", [(b.name, b.target_commit[:8]) for b in refs.branches])
    except Exception as e:
        print("  branches: ERRO", type(e).__name__)
    return info

print("=========== PARTE 0: bird1-think ANTES ===========")
describe(BIRD1, "(antes)")

print("=========== PARTE 0: tornando publico ===========")
if hasattr(api, "update_repo_settings"):
    api.update_repo_settings(repo_id=BIRD1, private=False)
else:
    api.update_repo_visibility(repo_id=BIRD1, private=False)
print("update_repo_settings executado")

print("=========== PARTE 0: bird1-think DEPOIS ===========")
describe(BIRD1, "(depois)")

print("=========== PARTE 1: repos a auditar ===========")
TARGETS = [
    ("cscsql-3b", "cycloneboy/CscSQL-Grpo-Qwen2.5-Coder-3B-Instruct"),
    ("gold-3b", "anonymous/sql-zero-3b-gold-control"),
    ("g1b-s78", "anonymous/sql-zero-3b-solver-bird1b-think"),
    ("bird2-s78", "anonymous/sql-zero-3b-solver-bird2-think"),
    ("bird3-s78", "anonymous/sql-zero-3b-solver-bird3-think"),
    ("g3-iter1-omni", "anonymous/sql-zero-7b-solver-omnisql-bird"),
    ("g3-iter2-omni", "anonymous/sql-zero-7b-solver-omnisql-bird-iter2"),
    ("g3-iter3-omni", "anonymous/sql-zero-7b-solver-omnisql-bird-iter3"),
]
for tag, repo in TARGETS:
    describe(repo, f"[{tag}]")

print("=========== PARTE 1: inventario completo do autor ===========")
for m in api.list_models(author="anonymous"):
    full = api.model_info(m.id)
    print(f"AUTHOR_REPO {m.id} private={full.private} usedStorage={getattr(full,'usedStorage',None)} lastmod={full.lastModified}")
print("AUDIT-COMPLETO")
PYEOF
