#!/bin/bash
#SBATCH --job-name=coverage-desc
#SBATCH --partition=gpu-node-a
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
set -euo pipefail
cd /workspace/anon/projects/dr-zero-t2sql
singularity exec --bind /workspace/anon:/workspace --pwd /workspace/projects/dr-zero-t2sql   --env PYTHONNOUSERSITE=1 /workspace/anon/images/verl0.5-sglang048.sif   python3 scripts/coverage_descriptive.py
echo COVERAGE_DONE
