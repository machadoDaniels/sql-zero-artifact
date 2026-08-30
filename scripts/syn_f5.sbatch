#!/bin/bash
#SBATCH --job-name=eval-syn-f5
#SBATCH --partition=gpu-node-b
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/workspace/anon/projects/dr-zero-t2sql/logs/%x-%j.log
singularity exec --nv --bind /workspace/anon:/workspace --bind /dev/shm:/dev/shm --pwd /workspace/projects/dr-zero-t2sql --env HF_HOME=/workspace/hf-cache --env PYTHONNOUSERSITE=1 --env SLURM_JOB_ID=$SLURM_JOB_ID /workspace/anon/images/vllm_openai_latest.sif bash scripts/syn_f5_inner.sh
