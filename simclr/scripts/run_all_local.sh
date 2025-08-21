#!/usr/bin/env bash
set -euo pipefail

MODES=(
  DINO_PRETRAINED_EVAL
  RANDOM_INIT_EVAL
  SCRATCH_IMAGENET_SIMCLR_PRETRAIN
  SCRATCH_IMAGENET_SUPERVISED_PRETRAIN
  DINO_IMAGENET_SIMCLR_FINETUNE
  DINO_IMAGENET_SUPERVISED_FINETUNE
  DINO_CIFAR_SUPERVISED_FINETUNE
  SCRATCH_CIFAR_SUPERVISED_TRAIN
)

for MODE in ${MODES[@]}; do
  echo "Running $MODE locally..."
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
 done
