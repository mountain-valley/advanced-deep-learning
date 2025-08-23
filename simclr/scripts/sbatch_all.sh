#!/usr/bin/env bash

set -euo pipefail

mkdir -p slurm_logs

MODES=(
  pretrained
  scratch
  simclr_scratch
  supervised_scratch
  simclr_pretrained
  supervised_pretrained
  cifar_supervised_pretrained
  cifar_supervised_scratch
)

SLEEP_BETWEEN="${SLEEP_BETWEEN:-0.2}"

# Desired max_steps for all jobs (-1 means run full epochs). Override by exporting MAX_STEPS before calling this script.
export MAX_STEPS="${MAX_STEPS:-10000}"

for MODE in "${MODES[@]}"; do
  echo "[LAUNCH] --mode=$MODE (MAX_STEPS=$MAX_STEPS)"
  sbatch --job-name "$MODE" scripts/sbatch_one.sh --mode "$MODE" --run-name "$MODE"
  sleep "$SLEEP_BETWEEN"
done

echo "All jobs submitted."
