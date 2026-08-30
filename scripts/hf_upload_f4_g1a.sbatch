#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
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
    ("anonymous/sql-zero-7b-solver-iter1-think", "step-75",  f"{CK}/t2sql_solver_iter1_7b_think_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_75"),
    ("anonymous/sql-zero-7b-solver-iter1-think", "step-200", f"{CK}/t2sql_solver_iter1_7b_think_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_200"),
    ("anonymous/sql-zero-7b-solver-iter1-think", "main",     f"{CK}/t2sql_solver_iter1_7b_think_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_240"),
    ("anonymous/sql-zero-3b-solver-bird1a",      "main",     f"{CK}/t2sql_solver_bird1a_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_75"),
    ("anonymous/sql-zero-3b-solver-bird1a",      "step-100", f"{CK}/t2sql_solver_bird1a_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_100"),
]
for repo, rev, path in jobs:
    assert os.path.isfile(os.path.join(path, "config.json")), f"config.json ausente em {path}"
    api.create_repo(repo, exist_ok=True, private=True)
    if rev != "main":
        try: api.create_branch(repo, branch=rev, exist_ok=True)
        except Exception as e: print("branch:", e)
    print(f"subindo {path} -> {repo}@{rev}")
    api.upload_folder(folder_path=path, repo_id=repo, revision=rev, commit_message=f"upload {os.path.basename(path)} (dr-zero-t2sql)")
    print(f"OK {repo}@{rev}")
print("UPLOADS COMPLETOS")
PYEOF
