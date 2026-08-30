#!/bin/bash
#SBATCH --job-name=hf-upload-g3-iter23
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.6-cu128.sif"
$W python - << "PYEOF"
import os
from huggingface_hub import HfApi
api = HfApi()
jobs = [
    ("anonymous/sql-zero-7b-solver-omnisql-bird-iter2", "checkpoints/dr-zero-t2sql/t2sql_solver_bird2_omni20k_grpo_group8_omnisql-7b/hf_model_78"),
    ("anonymous/sql-zero-7b-solver-omnisql-bird-iter3", "checkpoints/dr-zero-t2sql/t2sql_solver_bird3_omni20k_grpo_group8_omnisql-7b/hf_model_65"),
]
for repo, path in jobs:
    assert os.path.isfile(os.path.join(path, "config.json")), f"faltou {path}"
    api.create_repo(repo, exist_ok=True, private=True)
    print("subindo", path, "->", repo)
    api.upload_folder(folder_path=path, repo_id=repo, commit_message="checkpoint (dr-zero-t2sql, espiral BIRD G3)")
    print("OK", repo)
print("UPLOADS-G3-ITER23-COMPLETOS")
PYEOF
