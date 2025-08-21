#!/usr/bin/env bash
# Train SimCLR from scratch on (Tiny) ImageNet for ~1000 steps, then evaluate via linear probe on CIFAR-100.
# Defaults are offline-friendly and target Tiny-ImageNet; override via env vars below.
set -euo pipefail

# --- Configurable via env vars ---
STEPS="${STEPS:-5000}"                 # Max training steps for ImageNet SimCLR
IMAGE_SIZE="${IMAGE_SIZE:-64}"         # 64 for Tiny-ImageNet; use 224 for full ImageNet/backbones that require it
BATCH="${BATCH:-64}"                   # Pretrain batch size (tune to your GPU)
LR="${LR:-3e-5}"                        # Pretrain learning rate (defaults to main.py's default)
USE_TINY="${USE_TINY:-1}"               # 1 = use Tiny-ImageNet, 0 = use full ImageNet

# --- Paths ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/assets/datasets}"
TINY_IMAGENET_PATH="${TINY_IMAGENET_PATH:-${DATASET_ROOT}/tiny-imagenet/tiny-imagenet-200}"
IMAGENET_PATH="${IMAGENET_PATH:-${DATASET_ROOT}/imagenet}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

if [[ "${USE_TINY}" == "1" ]]; then
  echo "[INFO] Starting SCRATCH_IMAGENET_SIMCLR_PRETRAIN for ${STEPS} steps on Tiny-ImageNet at ${TINY_IMAGENET_PATH}"
else
  echo "[INFO] Starting SCRATCH_IMAGENET_SIMCLR_PRETRAIN for ${STEPS} steps on ImageNet at ${IMAGENET_PATH}"
fi
echo "[INFO] Image size: ${IMAGE_SIZE}, Batch: ${BATCH}, LR: ${LR}"

# This mode will: (1) pretrain SimCLR on (Tiny) ImageNet from scratch, (2) evaluate a linear head on CIFAR-100.
CMD=(
  uv run python -u "${ROOT_DIR}/main.py"
  --training_mode SCRATCH_IMAGENET_SIMCLR_PRETRAIN
  --dataset_root "${DATASET_ROOT}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --image_size "${IMAGE_SIZE}"
  --pretrain_batch_size "${BATCH}"
  --pretrain_lr "${LR}"
  --max_steps "${STEPS}"
)

if [[ "${USE_TINY}" == "1" ]]; then
  CMD+=(--use_tiny_imagenet --tiny_imagenet_path "${TINY_IMAGENET_PATH}")
else
  CMD+=(--imagenet_path "${IMAGENET_PATH}")
fi

"${CMD[@]}"

RESULTS_FILE="${OUTPUT_DIR}/results.log"
if [[ -f "${RESULTS_FILE}" ]]; then
  echo "[INFO] Done. Latest results in: ${RESULTS_FILE}"
  tail -n 5 "${RESULTS_FILE}" || true
else
  echo "[WARN] No results.log found at ${RESULTS_FILE}. Check the run logs under ${OUTPUT_DIR}/runs/."
fi
