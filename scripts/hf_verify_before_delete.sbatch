#!/bin/bash
#SBATCH --job-name=hf-verify-before-delete
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.6-cu128.sif"
$W python - << "PYEOF"
import os
from huggingface_hub import HfApi

api = HfApi()
CK = "checkpoints/dr-zero-t2sql"
PAIRS = [
    ("anonymous/sql-zero-7b-solver-bird2-iter2-think",
     f"{CK}/t2sql_solver_bird2x_7b20k_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_78"),
    ("anonymous/sql-zero-7b-solver-bird3-iter3-think",
     f"{CK}/t2sql_solver_bird3x_7b20k_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_78"),
]
for repo, path in PAIRS:
    info = api.repo_info(repo, files_metadata=True)
    remote = {s.rfilename: s.size for s in info.siblings}
    local = {f: os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
             if os.path.isfile(os.path.join(path, f))}
    print(f"=== {repo}  <->  {path}")
    missing = sorted(set(local) - set(remote))
    mismatch = sorted(f for f in set(local) & set(remote) if local[f] != remote[f])
    for f in sorted(local):
        r = remote.get(f)
        flag = "OK " if r == local[f] else ("AUSENTE" if r is None else "DIFERE")
        print(f"  {flag} {f}: local={local[f]} remoto={r}")
    print(f"  extras_no_remoto={sorted(set(remote)-set(local))}")
    print(f"  RESULTADO: missing={missing} mismatch={mismatch}")
    print(f"  VEREDITO: {'COMPLETO' if not missing and not mismatch else 'INCOMPLETO'}")
print("VERIFY-COMPLETO")
PYEOF
