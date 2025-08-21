#!/usr/bin/env bash
# Quick GPU smoke test: run SimCLR pretraining on (Tiny) ImageNet for 250 steps.
# Uses local offline assets prepared in ./assets.
set -euo pipefail

mkdir -p outputs

# Show GPU info if available
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi || true
fi

# Small conservative defaults; adjust as needed
IMAGE_SIZE=${IMAGE_SIZE:-64}                 # 64 for Tiny ImageNet
BATCH_SIZE=${BATCH_SIZE:-128}               # Reduce if OOM
LR=${LR:-5e-4}
TEMP=${TEMP:-0.2}
PROJ_DIM=${PROJ_DIM:-128}
MAX_STEPS=${MAX_STEPS:-250}
EPOCHS=${EPOCHS:-100}                        # Will stop by MAX_STEPS first

COMMON_ARGS=(
  --training_mode SCRATCH_IMAGENET_SIMCLR_PRETRAIN
  --dataset_root ./assets/datasets
  --use_tiny_imagenet
  --tiny_imagenet_path ./assets/datasets/tiny-imagenet/tiny-imagenet-200
  --output_dir ./outputs
  --model_name_or_path ./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m
  --image_size ${IMAGE_SIZE}
  --pretrain_batch_size ${BATCH_SIZE}
  --pretrain_lr ${LR}
  --temperature ${TEMP}
  --projection_dim ${PROJ_DIM}
  --pretrain_epochs ${EPOCHS}
  --max_steps ${MAX_STEPS}
)

if command -v uv >/dev/null 2>&1; then
  uv run python -u main.py "${COMMON_ARGS[@]}"
else
  python -u main.py "${COMMON_ARGS[@]}"
fi
