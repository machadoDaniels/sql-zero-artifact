#!/bin/bash
#SBATCH --job-name=hf-publish-bird1nt
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env XDG_CACHE_HOME=/workspace/tmp/xdg-cache /workspace/anon/images/verl0.6-cu128.sif"
$W python - << "PYEOF"
import hashlib
import os
import sys

from huggingface_hub import HfApi, hf_hub_download

api = HfApi()

# ---------------------------------------------------------------- helpers
def snapshot(repo, tag):
    info = api.repo_info(repo, files_metadata=True)
    files = {s.rfilename: s for s in info.siblings}
    shards = {k: v.size for k, v in files.items() if k.endswith(".safetensors")}
    snap = dict(
        private=info.private,
        sha=info.sha,
        usedStorage=getattr(info, "usedStorage", None),
        n_shards=len(shards),
        shard_bytes=sum(v or 0 for v in shards.values()),
    )
    print(f"REPO {repo} {tag}")
    for k, v in snap.items():
        print(f"    {k} = {v}")
    for k in sorted(shards):
        print(f"    shard {k} = {shards[k]}")
    return snap, files


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    size = os.path.getsize(path)
    h = hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def lfs_sha256(sib):
    lfs = getattr(sib, "lfs", None)
    if lfs is None:
        return None
    if isinstance(lfs, dict):
        return lfs.get("sha256")
    return getattr(lfs, "sha256", None)


# ============================================================ ACAO 1
print("=========== ACAO 1: tornar publicos 3 repos ===========")
TO_OPEN = [
    ("g1b-s78", "anonymous/sql-zero-3b-solver-bird1b-think"),
    ("bird2-s78", "anonymous/sql-zero-3b-solver-bird2-think"),
    ("g3-iter2-omni", "anonymous/sql-zero-7b-solver-omnisql-bird-iter2"),
]
problems = []
for tag, repo in TO_OPEN:
    print(f"----- [{tag}] {repo}")
    before, _ = snapshot(repo, "(antes)")
    api.update_repo_settings(repo_id=repo, private=False)
    print("    update_repo_settings(private=False) executado")
    after, _ = snapshot(repo, "(depois)")
    for field in ("sha", "usedStorage", "n_shards", "shard_bytes"):
        if before[field] != after[field]:
            problems.append(f"{repo}: {field} mudou {before[field]} -> {after[field]}")
    if after["private"] is not False:
        problems.append(f"{repo}: continua private={after['private']}")
    print(f"    VEREDITO [{tag}]: private {before['private']} -> {after['private']}, "
          f"sha {'IGUAL' if before['sha'] == after['sha'] else 'MUDOU'}, "
          f"usedStorage {'IGUAL' if before['usedStorage'] == after['usedStorage'] else 'MUDOU'}")

if problems:
    print("ACAO1-FALHOU")
    for p in problems:
        print("  PROBLEMA:", p)
    sys.exit(1)
print("ACAO1-OK: mudanca puramente de metadados nos 3 repos")

# ============================================================ ACAO 2
print("=========== ACAO 2: upload publico do bird1nt20k ===========")
REPO = "anonymous/sql-zero-3b-solver-bird1-nothink"
PATH = "checkpoints/dr-zero-t2sql/t2sql_solver_bird1nt20k_grpo_group8_qwen2.5-coder-3b-instruct/hf_model_78"

assert os.path.isfile(os.path.join(PATH, "config.json")), f"config.json ausente em {PATH}"
local_files = sorted(
    f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))
)
print("arquivos locais:", local_files)

try:
    info = api.repo_info(REPO)
    print(f"ATENCAO: {REPO} JA EXISTE (private={info.private}, sha={info.sha})")
    existed = True
except Exception as e:
    print(f"confirmado inexistente: {type(e).__name__}")
    existed = False

if not existed:
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=False)
    print("repo criado PUBLICO")

info = api.repo_info(REPO)
assert info.private is False, "repo nao esta publico; abortando upload"

print("subindo", PATH, "->", REPO)
api.upload_folder(
    folder_path=PATH,
    repo_id=REPO,
    commit_message="checkpoint bird1nt20k s78 (dr-zero-t2sql, braco no-think da iteracao 1, BIRD 3B)",
)
print("upload_folder concluido")

# ------------------------------------------------ verificacao byte a byte
print("----- verificacao pos-upload")
after_info = api.repo_info(REPO, files_metadata=True)
sibs = {s.rfilename: s for s in after_info.siblings}
print(f"private={after_info.private} sha={after_info.sha} "
      f"usedStorage={getattr(after_info, 'usedStorage', None)}")

fails = []
for name in local_files:
    lp = os.path.join(PATH, name)
    lsize = os.path.getsize(lp)
    sib = sibs.get(name)
    if sib is None:
        fails.append(f"{name}: AUSENTE no repo")
        continue
    if sib.size != lsize:
        fails.append(f"{name}: size {sib.size} != local {lsize}")
    remote_sha256 = lfs_sha256(sib)
    if remote_sha256 is not None:
        local_hash, kind, remote_hash = sha256_file(lp), "lfs-sha256", remote_sha256
    else:
        local_hash, kind, remote_hash = git_blob_sha1(lp), "git-blob-sha1", sib.blob_id
    ok = local_hash == remote_hash
    if not ok:
        fails.append(f"{name}: {kind} {remote_hash} != local {local_hash}")
    print(f"  {name}: size={lsize} {kind} {'OK' if ok else 'DIVERGE'}")

extra = sorted(set(sibs) - set(local_files) - {".gitattributes"})
if extra:
    print("  arquivos extras no repo (nao locais):", extra)

# download real dos arquivos pequenos e diff byte a byte
for small in ("config.json", "model.safetensors.index.json"):
    if small not in sibs:
        fails.append(f"{small}: ausente no repo")
        continue
    got = hf_hub_download(REPO, small, cache_dir="/workspace/tmp/hfverify")
    same = open(got, "rb").read() == open(os.path.join(PATH, small), "rb").read()
    print(f"  download+diff {small}: {'IDENTICO' if same else 'DIFERENTE'}")
    if not same:
        fails.append(f"{small}: conteudo baixado difere do local")

if fails:
    print("ACAO2-FALHOU")
    for f in fails:
        print("  PROBLEMA:", f)
    sys.exit(1)

print("ACAO2-OK: todos os arquivos casam byte a byte (sha256 LFS / git-blob-sha1)")
print("HF-PUBLISH-COMPLETO")
PYEOF
