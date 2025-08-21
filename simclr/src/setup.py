#!/usr/bin/env python3
"""Simple asset bootstrap script.

Downloads:
  - CIFAR-100 (train + test) into ./data/cifar100
  - Tiny ImageNet (official zip) into ./data/tiny-imagenet-200
  - DINOv3 model snapshot into ./models/dinov3-vits16-pretrain-lvd1689m

Run from any directory; paths are relative to the current working directory
where you invoke the script (not the script's location). Safe to re-run; it
skips work if targets already exist.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from dotenv import load_dotenv
import csv

import torchvision
from huggingface_hub import snapshot_download


TINY_IMAGENET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"
DEFAULT_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_cifar100(data_dir: Path) -> None:
    target = data_dir / "cifar100"
    ensure_dir(target)
    # Torchvision will skip if already present
    print(f"[CIFAR100] Downloading (if needed) to {target} ...")
    torchvision.datasets.CIFAR100(root=str(target), train=True, download=True)
    torchvision.datasets.CIFAR100(root=str(target), train=False, download=True)
    print("[CIFAR100] Ready.")


def normalize_tiny_imagenet_val(val_dir: Path) -> None:
    ann = val_dir / "val_annotations.txt"
    images = val_dir / "images"
    if not (ann.is_file() and images.is_dir()):
        return
    mapping: dict[str, str] = {}
    with ann.open("r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    for fname, cls in mapping.items():
        src = images / fname
        if not src.is_file():
            continue
        cls_dir = val_dir / cls
        cls_dir.mkdir(exist_ok=True)
        dst = cls_dir / fname
        if not dst.exists():
            shutil.move(str(src), str(dst))
    # remove empty images dir
    try:
        if images.exists() and not any(images.iterdir()):
            images.rmdir()
    except Exception:
        pass


def download_tiny_imagenet(data_dir: Path) -> Path:
    target_root = data_dir / "tiny-imagenet-200"
    if (target_root / "train").is_dir() and (target_root / "val").is_dir():
        print(f"[TinyImageNet] Already present at {target_root}")
        return target_root
    ensure_dir(data_dir)
    print(f"[TinyImageNet] Downloading zip to memory from {TINY_IMAGENET_URL} ... (≈ 244MB)")
    with urllib.request.urlopen(TINY_IMAGENET_URL) as resp:
        data = resp.read()
    print("[TinyImageNet] Extracting ...")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(path=str(data_dir))
    if not target_root.is_dir():
        raise RuntimeError("tiny-imagenet-200 directory missing after extraction")
    normalize_tiny_imagenet_val(target_root / "val")
    print(f"[TinyImageNet] Ready at {target_root}")
    return target_root


def download_model(models_dir: Path, model_id: str) -> Path:
    local_dir = models_dir / model_id.split('/')[-1]
    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[Model] Reusing existing snapshot at {local_dir}")
        return local_dir
    ensure_dir(local_dir)
    print(f"[Model] Downloading HF snapshot: {model_id} -> {local_dir}")
    snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    print("[Model] Ready.")
    return local_dir


def load_env() -> None:
    """Load environment variables from a .env file if present.

    Order:
      1. .env in the current working directory
      2. .env alongside this script (if different)
    Existing environment variables are not overwritten.
    """
    cwd_env = Path.cwd() / ".env"
    script_env = Path(__file__).resolve().parent / ".env"
    loaded_any = False
    if cwd_env.is_file():
        load_dotenv(dotenv_path=cwd_env, override=False)
        print(f"[Env] Loaded variables from {cwd_env}")
        loaded_any = True
    if script_env.is_file() and script_env != cwd_env:
        load_dotenv(dotenv_path=script_env, override=False)
        print(f"[Env] Loaded variables from {script_env}")
        loaded_any = True
    if not loaded_any:
        print("[Env] No .env file found (cwd or script directory). Proceeding without it.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CIFAR-100, Tiny ImageNet, and a DINOv3 model.")
    parser.add_argument("--data-dir", default="./data", help="Directory to place datasets (default: ./data)")
    parser.add_argument("--models-dir", default="./models", help="Directory to place model snapshots (default: ./models)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"HF model id (default: {DEFAULT_MODEL_ID})")
    return parser.parse_args()


def main() -> int:
    load_env()
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    ensure_dir(data_dir)
    ensure_dir(models_dir)

    download_cifar100(data_dir)
    download_tiny_imagenet(data_dir)
    download_model(models_dir, args.model_id)
    print("All assets ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
