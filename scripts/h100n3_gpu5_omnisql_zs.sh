#!/bin/bash
# OmniSQL-7B zero-shot prompt unificado: Spider dev + BIRD dev
# GPU 5 gpu-node-a FORA do Slurm (autorizado 2026-07-15, compartilhada c/ processo do joaopedro)
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
export CUDA_VISIBLE_DEVICES=5
PORT=23505
W="singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env CUDA_VISIBLE_DEVICES=5 --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.5-sglang048.sif"
$W bash -c "
set -x
python -m sglang.launch_server --model=/workspace/hf-cache/hub/models--seeklhy--OmniSQL-7B/snapshots/af4eed67f561bbeea555c017dae4b38b93bac2eb --port=$PORT --mem-fraction-static 0.75 --log-level=error &
SPID=\$!
for i in \$(seq 1 360); do curl -s http://127.0.0.1:$PORT/health > /dev/null && break; sleep 5; done
curl -s http://127.0.0.1:$PORT/health > /dev/null || { echo SERVER-TIMEOUT; exit 1; }
python scripts/eval_spider.py --dev_json ./spider/dev.json --db_root ./spider/database --endpoint http://127.0.0.1:$PORT/v1 --model /workspace/hf-cache/hub/models--seeklhy--OmniSQL-7B/snapshots/af4eed67f561bbeea555c017dae4b38b93bac2eb --out ./results/omnisql_zs_unified_spiderdev.json --concurrency 48
python scripts/eval_spider.py --dev_json ./bird/dev_20240627/dev.json --db_root ./bird/dev_databases --exec_timeout 30 --endpoint http://127.0.0.1:$PORT/v1 --model /workspace/hf-cache/hub/models--seeklhy--OmniSQL-7B/snapshots/af4eed67f561bbeea555c017dae4b38b93bac2eb --out ./results/omnisql_zs_unified_bird.json --concurrency 48
kill \$SPID 2>/dev/null || true
echo COMPLETO-GPU5"
