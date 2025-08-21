#!/usr/bin/env bash
# Submit a single job to Slurm for one training mode.
# Usage: sbatch scripts/sbatch_one.sh SCRATCH_IMAGENET_SIMCLR_PRETRAIN
# With new layout: datasets in ./data (tiny-imagenet-200, cifar100), models in ./models

#SBATCH --job-name=simclr_one
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

MODE=${1:?"Training mode is required"}
RUN_NAME=${RUN_NAME:-${MODE}}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"          # contains cifar100/ tiny-imagenet-200/
IMAGENET_PATH="${IMAGENET_PATH:-${DATA_DIR}/tiny-imagenet-200}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/dinov3-vits16-pretrain-lvd1689m}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

mkdir -p slurm_logs "${OUTPUT_DIR}"

echo "[SLURM] Mode=${MODE} RunName=${RUN_NAME} DataRoot=${DATA_DIR} TinyImagenet=${IMAGENET_PATH} Model=${MODEL_PATH}" 

PY_ARGS=(
  --training_mode "${MODE}"
  --dataset_root "${DATA_DIR}"
  --imagenet_path "${IMAGENET_PATH}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
)

if command -v uv >/dev/null 2>&1; then
  uv run python -u "${ROOT_DIR}/main.py" "${PY_ARGS[@]}"
else
  python -u "${ROOT_DIR}/main.py" "${PY_ARGS[@]}"
fi
