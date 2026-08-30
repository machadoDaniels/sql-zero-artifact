#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.5-sglang048.sif"
$W python - << "PYEOF"
import os
from huggingface_hub import HfApi
api = HfApi()
CK = "checkpoints/dr-zero-t2sql"
jobs = [
    ("anonymous/sql-zero-3b-solver-bird1b-think", "main", f"{CK}/t2sql_solver_bird1b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78"),
    ("anonymous/sql-zero-3b-solver-bird2-think",  "main", f"{CK}/t2sql_solver_bird2b20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78"),
]
for repo, rev, path in jobs:
    assert os.path.isfile(os.path.join(path, "config.json")), f"faltou {path}"
    api.create_repo(repo, exist_ok=True, private=True)
    print("subindo", path, "->", repo)
    api.upload_folder(folder_path=path, repo_id=repo, revision=rev, commit_message=f"upload {os.path.basename(path)} (espiral BIRD think)")
    print("OK", repo)
print("UPLOADS COMPLETOS")
PYEOF
