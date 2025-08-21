#!/usr/bin/env bash
# Quick smoke test for all modes with minimal steps/epochs.
# Assumes offline assets under ./assets and Tiny-ImageNet prepared.
set -euo pipefail

mkdir -p outputs

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi || true
fi

# Defaults tuned for speed and compatibility with ViT backbones
IMAGE_SIZE=${IMAGE_SIZE:-224}
IMAGENET_STEPS=${IMAGENET_STEPS:-100}
CIFAR_STEPS=${CIFAR_STEPS:-200}
PROBE_EPOCHS=${PROBE_EPOCHS:-1}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-1}
PRETRAIN_LR=${PRETRAIN_LR:-5e-4}
FINETUNE_LR=${FINETUNE_LR:-1e-4}
PROBE_LR=${PROBE_LR:-1e-3}
PRETRAIN_BS=${PRETRAIN_BS:-32}
FINETUNE_BS=${FINETUNE_BS:-64}
TEMP=${TEMP:-0.2}
PROJ_DIM=${PROJ_DIM:-128}

COMMON_BASE=(
  --dataset_root ./assets/datasets
  --use_tiny_imagenet
  --tiny_imagenet_path ./assets/datasets/tiny-imagenet/tiny-imagenet-200
  --output_dir ./outputs
  --model_name_or_path ./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m
  --image_size ${IMAGE_SIZE}
)

run() {
  echo "\n===== RUN: $* =====\n"
  if command -v uv >/dev/null 2>&1; then
    uv run python -u main.py "$@"
  else
    python -u main.py "$@"
  fi
}

# 1) DINO pretrained linear eval (fast)
run \
  --training_mode DINO_PRETRAINED_EVAL \
  "${COMMON_BASE[@]}" \
  --probe_epochs ${PROBE_EPOCHS} \
  --probe_lr ${PROBE_LR} \
  --max_steps -1

# 2) Random init linear eval (fast)
run \
  --training_mode RANDOM_INIT_EVAL \
  "${COMMON_BASE[@]}" \
  --probe_epochs ${PROBE_EPOCHS} \
  --probe_lr ${PROBE_LR} \
  --max_steps -1

# 3) SimCLR pretrain on Tiny-ImageNet (short)
run \
  --training_mode SCRATCH_IMAGENET_SIMCLR_PRETRAIN \
  "${COMMON_BASE[@]}" \
  --pretrain_batch_size ${PRETRAIN_BS} \
  --pretrain_lr ${PRETRAIN_LR} \
  --temperature ${TEMP} \
  --projection_dim ${PROJ_DIM} \
  --pretrain_epochs 20 \
  --max_steps ${IMAGENET_STEPS}

# 4) Supervised pretrain on Tiny-ImageNet (short)
run \
  --training_mode SCRATCH_IMAGENET_SUPERVISED_PRETRAIN \
  "${COMMON_BASE[@]}" \
  --pretrain_batch_size ${PRETRAIN_BS} \
  --pretrain_lr ${PRETRAIN_LR} \
  --pretrain_epochs 20 \
  --max_steps ${IMAGENET_STEPS}

# 5) Finetune (SimCLR) starting from pretrained DINO (short)
run \
  --training_mode DINO_IMAGENET_SIMCLR_FINETUNE \
  "${COMMON_BASE[@]}" \
  --pretrain_batch_size ${PRETRAIN_BS} \
  --pretrain_lr ${PRETRAIN_LR} \
  --temperature ${TEMP} \
  --projection_dim ${PROJ_DIM} \
  --pretrain_epochs 20 \
  --max_steps ${IMAGENET_STEPS}

# 6) Finetune (Supervised) starting from pretrained DINO (short)
run \
  --training_mode DINO_IMAGENET_SUPERVISED_FINETUNE \
  "${COMMON_BASE[@]}" \
  --pretrain_batch_size ${PRETRAIN_BS} \
  --pretrain_lr ${PRETRAIN_LR} \
  --pretrain_epochs 20 \
  --max_steps ${IMAGENET_STEPS}

# 7) CIFAR supervised finetune (short)
run \
  --training_mode DINO_CIFAR_SUPERVISED_FINETUNE \
  "${COMMON_BASE[@]}" \
  --finetune_batch_size ${FINETUNE_BS} \
  --finetune_lr ${FINETUNE_LR} \
  --finetune_epochs ${FINETUNE_EPOCHS} \
  --max_steps ${CIFAR_STEPS}

# 8) CIFAR supervised from scratch (short)
run \
  --training_mode SCRATCH_CIFAR_SUPERVISED_TRAIN \
  "${COMMON_BASE[@]}" \
  --finetune_batch_size ${FINETUNE_BS} \
  --finetune_lr ${FINETUNE_LR} \
  --finetune_epochs ${FINETUNE_EPOCHS} \
  --max_steps ${CIFAR_STEPS}

echo "\nAll quick tests completed. Logs in ./outputs and TensorBoard runs under ./outputs/runs."
