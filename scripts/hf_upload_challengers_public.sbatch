#!/bin/bash
#SBATCH --job-name=hf-upload-challengers-public
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
echo "QUOTA-ANTES: $(du -sh /workspace/anon | cut -f1)"
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache --env HF_HUB_DISABLE_XET=1 /workspace/anon/images/verl0.6-cu128.sif"
$W python - << "PYEOF"
import os
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

api = HfApi()
CK = "checkpoints/dr-zero-t2sql"
PAIRS = [
    ("anonymous/sql-zero-7b-challenger-bird1-omni",
     f"{CK}/t2sql_challenger_bird1_omni_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model"),
    ("anonymous/sql-zero-7b-challenger-bird3-omni",
     f"{CK}/t2sql_challenger_bird3_omni_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model"),
    ("anonymous/sql-zero-7b-challenger-bird3x",
     f"{CK}/t2sql_challenger_bird3x_7b_grpo_batch_group1-5_qwen2.5-coder-7b-instruct/hf_model"),
]

# 1) pre-check: merge local completo + repo id ainda nao existe (nunca sobrescrever)
for repo, path in PAIRS:
    assert os.path.isfile(os.path.join(path, "config.json")), f"faltou config.json em {path}"
    assert os.path.isfile(os.path.join(path, "model.safetensors.index.json")), f"faltou index em {path}"
    n = len([f for f in os.listdir(path) if f.endswith(".safetensors")])
    assert n == 4, f"{path}: esperava 4 shards, achei {n}"
    try:
        api.repo_info(repo)
        raise SystemExit(f"ABORTADO: repo {repo} JA EXISTE; nao sobrescrever")
    except RepositoryNotFoundError:
        print("id livre:", repo)

# 2) criar publico desde o inicio e subir
for repo, path in PAIRS:
    api.create_repo(repo, exist_ok=False, private=False)
    info = api.repo_info(repo)
    print("visibilidade privada?", info.private)
    assert info.private is False, f"{repo} nasceu privado; abortando"
    print("subindo", path, "->", repo, flush=True)
    api.upload_folder(
        folder_path=path,
        repo_id=repo,
        commit_message="challenger checkpoint (dr-zero-t2sql, self-play BIRD)",
    )
    print("OK", repo, flush=True)

# 3) verificacao byte a byte
ok = True
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
    veredito = "COMPLETO" if not missing and not mismatch else "INCOMPLETO"
    ok = ok and veredito == "COMPLETO"
    print(f"  VEREDITO: {veredito}")
assert ok, "algum repo ficou INCOMPLETO"
print("UPLOAD-CHALLENGERS-PUBLIC-COMPLETO")
PYEOF
echo "QUOTA-DEPOIS: $(du -sh /workspace/anon | cut -f1)"
