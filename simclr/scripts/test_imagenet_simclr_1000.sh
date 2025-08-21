#!/usr/bin/env bash
# SimCLR pretraining on tiny-imagenet-200 (as the default "ImageNet" stand‑in) + CIFAR-100 linear probe.
# Updated to match new project layout (./data, ./models) and removal of optional Tiny/full ImageNet switch.
set -euo pipefail

# --- Configurable via env vars ---
STEPS="${STEPS:-1000}"                 # Max training steps for SimCLR pretraining
IMAGE_SIZE="${IMAGE_SIZE:-64}"         # 64 works for tiny-imagenet-200; increase if you upsample
BATCH="${BATCH:-64}"                   # Pretrain batch size
LR="${LR:-3e-5}"                        # Pretrain learning rate
RUN_NAME="${RUN_NAME:-simclr_imagenet_5000}"    # Name appended in TensorBoard run directory

# --- Paths (new layout) ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${ROOT_DIR}/data}"
IMAGENET_PATH="${IMAGENET_PATH:-${DATASET_ROOT}/tiny-imagenet-200}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/dinov3-vits16-pretrain-lvd1689m}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

echo "[INFO] Starting SIMCLR_SCRATCH for ${STEPS} steps on tiny-imagenet-200 at ${IMAGENET_PATH}"
echo "[INFO] Image size: ${IMAGE_SIZE}, Batch: ${BATCH}, LR: ${LR}, Run: ${RUN_NAME}"

# Command
uv run python -u "${ROOT_DIR}/src/main.py" \
  --training_mode "simclr_scratch" \
  --dataset_root "${DATASET_ROOT}" \
  --imagenet_path "${IMAGENET_PATH}" \
  --model_name_or_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --image_size "${IMAGE_SIZE}" \
  --pretrain_batch_size "${BATCH}" \
  --pretrain_lr "${LR}" \
  --max_steps "${STEPS}" \
  --run-name "${RUN_NAME}" \
  "$@"

RESULTS_FILE="${OUTPUT_DIR}/results.log"
if [[ -f "${RESULTS_FILE}" ]]; then
  echo "[INFO] Done. Latest results in: ${RESULTS_FILE}"
  tail -n 5 "${RESULTS_FILE}" || true
else
  echo "[WARN] No results.log found at ${RESULTS_FILE}. Check the run logs under ${OUTPUT_DIR}/runs/."
fi
