#!/bin/bash
#SBATCH --job-name=mcnemar-think-nt
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
W="singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.5-sglang048.sif"
$W python /workspace/projects/dr-zero-t2sql/scripts/mcnemar_think_nothink.py
