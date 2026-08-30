#!/bin/bash
#SBATCH --job-name=hf-availability-audit2
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


TARGETS = [
    ("bird2nt-s78", "anonymous/sql-zero-3b-solver-bird2-nothink"),
    ("bird3nt-s78", "anonymous/sql-zero-3b-solver-bird3-nothink"),
    ("bird1nt-s78?", "anonymous/sql-zero-3b-solver-bird1-nothink"),
    ("bird1nt-s78?", "anonymous/sql-zero-3b-solver-bird1nt-think"),
    ("bird1nt-s78?", "anonymous/sql-zero-3b-solver-bird1nt20k"),
]
for tag, repo in TARGETS:
    describe(repo, f"[{tag}]")
print("AUDIT2-COMPLETO")
PYEOF
