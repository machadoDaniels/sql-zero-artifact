#!/bin/bash
#SBATCH --job-name=gen-bird
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env SGLANG_ATTENTION_BACKEND=flashinfer --env SLURM_JOB_ID=$SLURM_JOB_ID /workspace/anon/images/verl0.6-cu128.sif bash scripts/gen_bird_inner.sh
