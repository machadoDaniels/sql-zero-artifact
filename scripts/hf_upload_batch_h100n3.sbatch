#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.5-sglang048.sif"
$W python - << "PYEOF"
import os
from huggingface_hub import HfApi
api = HfApi()
jobs = [
    ("anonymous/sql-zero-7b-challenger-bird1", "/workspace/tmp/upload-ch-bird1-7b"),
    ("anonymous/sql-zero-3b-challenger-bird2", "/workspace/tmp/upload-ch-bird2-3b"),
    ("anonymous/sql-zero-7b-challenger-iter1", "checkpoints/dr-zero-t2sql/t2sql_challenger_iter1_7b_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model"),
]
for repo, path in jobs:
    assert os.path.isfile(os.path.join(path, "config.json")), f"faltou {path}"
    api.create_repo(repo, exist_ok=True, private=True)
    print("subindo", path, "->", repo)
    api.upload_folder(folder_path=path, repo_id=repo, commit_message="challenger checkpoint (dr-zero-t2sql)")
    print("OK", repo)
print("UPLOADS-H100N3-COMPLETOS")
PYEOF
