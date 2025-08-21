#!/usr/bin/env bash
# Submit a single job to Slurm for one training mode.
# Usage: sbatch scripts/sbatch_one.sh DINO_PRETRAINED_EVAL

#SBATCH --job-name=simclr_one
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

MODE=${1:?"Training mode is required"}

mkdir -p slurm_logs outputs

# Activate uv environment if available; otherwise rely on system python
if command -v uv >/dev/null 2>&1; then
  uv run python -u main.py \
    --training_mode "$MODE" \
    --dataset_root ./assets/datasets \
    --use_tiny_imagenet \
    --tiny_imagenet_path ./assets/datasets/tiny-imagenet/tiny-imagenet-200 \
    --output_dir ./outputs \
    --model_name_or_path ./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m
else
  python -u main.py \
    --training_mode "$MODE" \
    --dataset_root ./assets/datasets \
    --use_tiny_imagenet \
    --tiny_imagenet_path ./assets/datasets/tiny-imagenet/tiny-imagenet-200 \
    --output_dir ./outputs \
    --model_name_or_path ./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m
fi
