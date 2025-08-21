#!/usr/bin/env bash
# Submit all training/eval modes as independent Slurm jobs (run on login node).
set -euo pipefail

mkdir -p slurm_logs

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
  sbatch scripts/sbatch_one.sh ${MODE}
  echo "Launched ${MODE}"
  sleep 0.2
done
