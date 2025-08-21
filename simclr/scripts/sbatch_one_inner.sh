#!/usr/bin/env bash
# Internal runner used by sbatch_all.sh to keep per-mode resources separate.
set -euo pipefail
MODE=${1:?"Training mode is required"}
mkdir -p slurm_logs outputs

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
