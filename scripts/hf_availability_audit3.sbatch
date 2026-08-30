#!/bin/bash
#SBATCH --job-name=hf-availability-audit3
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

CK = "checkpoints/dr-zero-t2sql"
LOCAL = {
    "ch_bird1_omni":    (f"{CK}/t2sql_challenger_bird1_omni_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model",
                         [4980423752, 4885039288, 4921667552, 444141216]),
    "ch_bird3_omni":    (f"{CK}/t2sql_challenger_bird3_omni_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model",
                         [4885002384, 4701616544, 4235347624, 1409305296]),
    "ch_bird3x_7b":     (f"{CK}/t2sql_challenger_bird3x_7b_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model",
                         [4921723544, 4892329248, 4954766112, 462452984]),
    "sol_bird2x_7b20k": (f"{CK}/t2sql_solver_bird2x_7b20k_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_78",
                         [4969441616, 4998739616, 4991509232, 271581448]),
    "sol_bird3x_7b20k": (f"{CK}/t2sql_solver_bird3x_7b20k_grpo_group8_qwen2.5-coder-7b-instruct/hf_model_78",
                         [4914380832, 4998797320, 4899671392, 418422328]),
}
CFG_SIZE = 1318
IDX_SIZE = 27788

def snapshot(repo, revision=None):
    info = api.repo_info(repo, revision=revision, files_metadata=True)
    files = {s.rfilename: s.size for s in info.siblings}
    shards = {k: v for k, v in files.items() if k.endswith(".safetensors")}
    return info, files, shards

print("=========== INVENTARIO COMPLETO (todos repos, todas revisoes) ===========")
snapshots = []
for m in api.list_models(author="anonymous"):
    repo = m.id
    try:
        refs = api.list_repo_refs(repo)
        branches = [b.name for b in refs.branches]
    except Exception as e:
        print(f"REPO {repo} :: list_repo_refs ERRO {type(e).__name__}")
        branches = ["main"]
    for br in branches:
        try:
            info, files, shards = snapshot(repo, br)
        except Exception as e:
            print(f"REPO {repo}@{br} :: ERRO {type(e).__name__}")
            continue
        print(f"REPO {repo}@{br}")
        print(f"  private={info.private} usedStorage={getattr(info,'usedStorage',None)} sha={info.sha} lastmod={info.lastModified}")
        print(f"  config.json={files.get('config.json')} index.json={files.get('model.safetensors.index.json')}")
        print(f"  n_shards={len(shards)} total_shard_bytes={sum(v or 0 for v in shards.values())}")
        for k in sorted(shards):
            print(f"    {k} = {shards[k]}")
        snapshots.append((repo, br, files, shards))

print("=========== MATCH LOCAL -> REMOTO (byte a byte) ===========")
for tag, (path, sizes) in LOCAL.items():
    print(f"--- {tag}  ({path})")
    print(f"    shards locais: {sizes}  soma={sum(sizes)}")
    found = False
    for repo, br, files, shards in snapshots:
        remote = [shards.get(f"model-0000{i}-of-00004.safetensors") for i in range(1, 5)]
        if remote == sizes and len(shards) == 4:
            ok_cfg = files.get("config.json") == CFG_SIZE
            ok_idx = files.get("model.safetensors.index.json") == IDX_SIZE
            print(f"    MATCH EXATO em {repo}@{br}  config_ok={ok_cfg} index_ok={ok_idx}")
            found = True
    if not found:
        print("    SEM MATCH em nenhum repo/revisao do autor")
print("AUDIT3-COMPLETO")
PYEOF
