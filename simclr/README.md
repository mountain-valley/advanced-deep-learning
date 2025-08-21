SimCLR and Supervised Training (offline-friendly, Slurm-ready)

This repo provides a minimal framework to pretrain/evaluate with a DINOv3 ViT-S backbone in offline HPC clusters using Slurm.

What's included
- main.py: training/eval entrypoint with modes
- scripts/setup_offline_assets.py: download CIFAR-100 and snapshot the HF model for offline use
- scripts/sbatch_one.sh: submit a single mode to Slurm
- scripts/sbatch_all.sh: submit all modes as independent jobs
- scripts/bootstrap_env.sh: optional uv-based environment bootstrap

Offline assets layout
- assets/datasets/
  - cifar100/ (created by setup)
  - imagenet/ (you must prepare with train/ and val/)
- assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m/ (created by setup)

Quick start (on a machine with internet)
1) Install uv and sync deps
   - uv sync
2) Prepare offline assets
   - uv run python scripts/setup_offline_assets.py \
       --dataset_root ./assets/datasets \
       --imagenet_path ./assets/datasets/imagenet \
       --model_id facebook/dinov3-vits16-pretrain-lvd1689m \
       --model_local_dir ./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m
3) Copy this whole directory to the offline compute node/filesystem

Run on Slurm
- Single mode:
  - sbatch scripts/sbatch_one.sh DINO_PRETRAINED_EVAL
- All modes:
  - sbatch scripts/sbatch_all.sh

Notes
- ImageNet is not downloaded by the setup script due to license constraints; point --imagenet_path to a prepared directory with train/ and val/.
- Defaults in main.py use the local snapshot of facebook/dinov3-vits16-pretrain-lvd1689m.
