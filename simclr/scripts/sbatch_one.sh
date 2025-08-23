#!/usr/bin/env bash

#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=dw87

# Fail on any error
set -euo pipefail

# Set root directory
# Use SLURM_SUBMIT_DIR if set else use the script directory's parent
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
fi

cd "${ROOT_DIR}"

# Argument Parsing
MODE=""
RUN_NAME=""
args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode=*) MODE="${1#*=}" ;;
    --mode) shift; MODE="${1:-}" ;;
    --run-name=*) RUN_NAME="${1#*=}" ;;
    --run-name) shift; RUN_NAME="${1:-}" ;;
    *) ;; # ignore unknown positional; could be legacy MODE value
  esac
  shift || true
done

if [[ -z "$MODE" ]]; then
  for a in "${args[@]}"; do
    if [[ "$a" != --* ]]; then
      MODE="$a"; break
    fi
  done
fi

# Usage Check
if [[ -z "$MODE" ]]; then
  echo "Usage: $0 --mode=simclr_scratch [--run-name=myrun]"
  echo "Or: $0 simclr_scratch"
  exit 1
fi

# Set Run Name if not provided
if [[ -z "$RUN_NAME" || "$RUN_NAME" == --* ]]; then
  RUN_NAME="$MODE"
fi

# Set Data Directories
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data}"
IMAGENET_PATH="${IMAGENET_PATH:-${DATA_DIR}/tiny-imagenet-200}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/dinov3-vits16-pretrain-lvd1689m}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs}"

# Create necessary directories
mkdir -p slurm_logs "${OUTPUT_DIR}"

# Log the configuration
echo "[SLURM] Mode=${MODE} RunName=${RUN_NAME} DataRoot=${DATA_DIR} TinyImagenet=${IMAGENET_PATH} Model=${MODEL_PATH}"
echo "[DIRECTORY] $(pwd)"

# Construct the Python arguments
PY_ARGS=(
  --training_mode "${MODE}"
  --dataset_root "${DATA_DIR}"
  --imagenet_path "${IMAGENET_PATH}"
  --model_name_or_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --max_steps "${MAX_STEPS:-1000}"
)

# Activate the environment
source "${ROOT_DIR}/.venv/bin/activate"

# Debugging
nvidia-smi
python - <<'PY'
import torch, os
print("Torch:", torch.__version__, "CUDA runtime:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device 0 name:", torch.cuda.get_device_name(0))
    cc = torch.cuda.get_device_capability(0)
    print("Compute capability:", cc)
    x = torch.randn(1, device='cuda')
    print("Simple op OK:", (x+x).item())
else:
    print("CUDA not available at runtime")
PY

# Run the main script
python -u "${ROOT_DIR}/src/main.py" "${PY_ARGS[@]}"
