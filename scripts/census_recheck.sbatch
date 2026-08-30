#!/bin/bash
#SBATCH --job-name=census-recheck
#SBATCH --partition=gpu-node-b
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env HF_HUB_OFFLINE=1 --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.6-cu128.sif"
$W python /workspace/projects/dr-zero-t2sql/scripts/census_recheck.py
