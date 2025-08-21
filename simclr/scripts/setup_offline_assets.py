#!/usr/bin/env python3
"""
Prepare offline assets: CIFAR-100 dataset and Hugging Face model repo snapshot.
Run this on a machine with internet (e.g., login node). ImageNet is NOT downloaded;
point --imagenet_path to an existing prepared directory with train/ and val/.
"""
import argparse
import os
from pathlib import Path

import shutil
import csv
import urllib.request
import zipfile
import io
import torchvision
from huggingface_hub import snapshot_download


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def download_cifar100(dataset_root: str):
    cifar_root = os.path.join(dataset_root, 'cifar100')
    ensure_dir(cifar_root)
    print(f"Downloading CIFAR-100 to: {cifar_root}")
    torchvision.datasets.CIFAR100(root=cifar_root, train=True, download=True)
    torchvision.datasets.CIFAR100(root=cifar_root, train=False, download=True)
    print("CIFAR-100 download complete.")


def download_hf_repo(model_id: str, target_dir: str):
    print(f"Snapshotting HF repo '{model_id}' to: {target_dir}")
    ensure_dir(target_dir)
    # Download all files for the repo. This works for models, configs, processors, etc.
    snapshot_download(repo_id=model_id, local_dir=target_dir, local_dir_use_symlinks=False)
    print("HF snapshot complete.")


def _normalize_tiny_imagenet_val(val_dir: str):
    """If val/ has images/ and val_annotations.txt, reorganize into class subfolders.
    This matches the typical Tiny ImageNet-200 packaging.
    """
    ann_path = os.path.join(val_dir, 'val_annotations.txt')
    images_dir = os.path.join(val_dir, 'images')
    if not (os.path.isfile(ann_path) and os.path.isdir(images_dir)):
        return
    # Build mapping: filename -> class
    mapping = {}
    with open(ann_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]
    # Create class subfolders and move images
    for fname, cls in mapping.items():
        src = os.path.join(images_dir, fname)
        if not os.path.isfile(src):
            continue
        cls_dir = os.path.join(val_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        dst = os.path.join(cls_dir, fname)
        if not os.path.exists(dst):
            shutil.move(src, dst)
    # Cleanup images folder if empty
    try:
        if not os.listdir(images_dir):
            os.rmdir(images_dir)
    except Exception:
        pass


def _download_tiny_imagenet_zip_to(target_dir: str) -> str:
    url = 'https://cs231n.stanford.edu/tiny-imagenet-200.zip'
    print(f"Downloading Tiny ImageNet from {url} (official source)...")
    ensure_dir(target_dir)
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(path=target_dir)
    base = os.path.join(target_dir, 'tiny-imagenet-200')
    if not os.path.isdir(base):
        raise RuntimeError('tiny-imagenet-200 folder missing after extraction')
    print("Tiny ImageNet zip extracted to:", base)
    return base


def download_tiny_imagenet(dataset_root: str, repo_id: str = 'SteveZeyuZhang/Tiny-ImageNet-200'):
    target_dir = os.path.join(dataset_root, 'tiny-imagenet')
    ensure_dir(target_dir)
    print(f"Snapshotting Tiny ImageNet repo '{repo_id}' (dataset) to: {target_dir}")
    local_path = None
    try:
        local_path = snapshot_download(repo_id=repo_id, repo_type='dataset', local_dir=target_dir, local_dir_use_symlinks=False)
        print("Tiny ImageNet snapshot complete at:", local_path)
    except Exception as e:
        print(f"[WARN] HF dataset snapshot failed for '{repo_id}': {e}\nFalling back to official Tiny ImageNet zip download.")
        return _download_tiny_imagenet_zip_to(target_dir)

    # Try to find tiny-imagenet-200 subdir
    base = target_dir
    candidate = os.path.join(target_dir, 'tiny-imagenet-200')
    if os.path.isdir(candidate):
        base = candidate

    train_dir = os.path.join(base, 'train')
    val_dir = os.path.join(base, 'val')
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print("[WARN] Tiny ImageNet train/val not found under:", base)
        # Try fallback zip extraction if we used HF dataset and structure isn't present
        try:
            base = _download_tiny_imagenet_zip_to(target_dir)
            train_dir = os.path.join(base, 'train')
            val_dir = os.path.join(base, 'val')
            if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
                print("[ERROR] Even after zip fallback, Tiny ImageNet structure is missing.")
                return None
        except Exception as e:
            print("[ERROR] Tiny ImageNet zip fallback failed:", e)
            return None
    _normalize_tiny_imagenet_val(val_dir)
    return base


def verify_imagenet(imagenet_path: str):
    train_dir = os.path.join(imagenet_path, 'train')
    val_dir = os.path.join(imagenet_path, 'val')
    if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
        print("[WARN] ImageNet not found at:", imagenet_path)
        print("       Expected subfolders: 'train/' and 'val/'. Skipping.")
        return False
    print("ImageNet directory detected:", imagenet_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare offline datasets and HF models")
    parser.add_argument('--dataset_root', default='./assets/datasets', help='Root for datasets')
    parser.add_argument('--imagenet_path', default='./assets/datasets/imagenet', help='Path to ImageNet (pre-prepared)')
    parser.add_argument('--model_id', default='facebook/dinov3-vits16-pretrain-lvd1689m', help='HF model repo id')
    parser.add_argument('--model_local_dir', default='./assets/hf_models/facebook/dinov3-vits16-pretrain-lvd1689m', help='Where to store the HF snapshot')
    parser.add_argument('--download_tiny_imagenet', action='store_true', help='Also download Tiny ImageNet and expose it as ImageNet locally')
    args = parser.parse_args()

    ensure_dir(args.dataset_root)
    download_cifar100(args.dataset_root)
    tiny_base = None
    if args.download_tiny_imagenet:
        tiny_base = download_tiny_imagenet(args.dataset_root)
        if tiny_base is not None:
            # Expose as assets/datasets/imagenet with train/ and val/
            imagenet_alias = args.imagenet_path
            ensure_dir(imagenet_alias)
            for split in ('train', 'val'):
                src = os.path.join(tiny_base, split)
                dst = os.path.join(imagenet_alias, split)
                # Prefer symlink; fallback to copy if needed
                if os.path.islink(dst) or os.path.exists(dst):
                    pass
                else:
                    try:
                        os.symlink(src, dst)
                    except OSError:
                        print(f"Symlink failed for {src} -> {dst}, copying instead (this may take time)...")
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
            print("Tiny ImageNet exposed at:", imagenet_alias)
    if not verify_imagenet(args.imagenet_path):
        print("Note: You can rerun this script with --download_tiny_imagenet to populate a small ImageNet-like dataset.")
    download_hf_repo(args.model_id, args.model_local_dir)
    print("All done. You can now run training offline.")


if __name__ == '__main__':
    main()
