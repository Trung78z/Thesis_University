#!/usr/bin/env python3
"""
Split images and YOLO‐format label txt files into **train**, **val**, and **test** folders.

Given a directory structure like

    dataset_root/
        images/
            aaa.jpg
            bbb.jpg
        labels/
            aaa.txt
            bbb.txt

this script creates (by default, next to the current working directory unless --outdir is given):

    dataset/
        data_split/
            train/
                images/
                labels/
            valid/              # (val)
                images/
                labels/
            test/
                images/
                labels/

Each image keeps its matching <file>.txt if it exists; otherwise the image is still copied and treated as background.

Defaults
--------
* **70 %** → train
* **20 %** → val
* **10 %** → test (the implicit remainder)

Examples
~~~~~~~~
Split 70 % train, 20 % val, 10 % test (defaults):

    python split_dataset_train_val_test.py \
        --datapath /data/my_yolo_dataset

Split 80 % train, 10 % val, 10 % test with reproducible shuffle:

    python split_dataset_train_val_test.py \
        --datapath /data/my_yolo_dataset \
        --train_pct 0.8 --val_pct 0.1 --seed 123
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split image/label data into train/val/test folders (YOLO format).",
    )
    parser.add_argument(
        "--datapath",
        required=True,
        help="Path to dataset root containing 'images' and 'labels' sub‑folders.",
    )
    parser.add_argument(
        "--train_pct",
        type=float,
        default=0.7,
        help="Fraction of images to allocate to the training set (default 0.7).",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=0.2,
        help="Fraction of images to allocate to the validation set (default 0.2). The remainder goes to test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Custom output directory; default creates 'dataset/data_split' in the CWD.",
    )
    return parser.parse_args()


def make_dirs(paths: List[Path]):
    for p in paths:
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Created directory: {p}")


def main():
    args = parse_args()

    data_path = Path(args.datapath).expanduser().resolve()
    if not data_path.exists():
        print("[ERROR] --datapath directory not found.")
        sys.exit(1)

    input_image_path = data_path / "images"
    input_label_path = data_path / "labels"

    if not input_image_path.exists() or not input_label_path.exists():
        print("[ERROR] 'images' or 'labels' directory missing inside datapath.")
        sys.exit(1)

    train_pct: float = args.train_pct
    val_pct: float = args.val_pct

    if train_pct <= 0 or val_pct < 0 or (train_pct + val_pct) >= 1:
        print(
            "[ERROR] train_pct must be > 0, val_pct must be >= 0, and their sum must be < 1.",
        )
        sys.exit(1)

    test_pct: float = 1.0 - train_pct - val_pct
    print(
        f"Splitting dataset → train: {train_pct:.2%}, val: {val_pct:.2%}, test: {test_pct:.2%}",
    )

    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed {args.seed} for reproducibility.")

    # Output structure
    base_out = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else Path.cwd() / "dataset" / "data_split"
    )

    train_img_path = base_out / "train" / "images"
    train_lbl_path = base_out / "train" / "labels"
    val_img_path = base_out / "valid" / "images"
    val_lbl_path = base_out / "valid" / "labels"
    test_img_path = base_out / "test" / "images"
    test_lbl_path = base_out / "test" / "labels"

    make_dirs(
        [
            train_img_path,
            train_lbl_path,
            val_img_path,
            val_lbl_path,
            test_img_path,
            test_lbl_path,
        ],
    )

    # Collect files
    img_files = sorted(input_image_path.rglob("*.*"))  # consider every file as potential image
    total_images = len(img_files)
    if total_images == 0:
        print("[ERROR] No images found under 'images' directory.")
        sys.exit(1)

    print(f"[INFO] Found {total_images} images. Starting split …")

    # Shuffle list
    random.shuffle(img_files)

    train_cutoff = int(total_images * train_pct)
    val_cutoff = train_cutoff + int(total_images * val_pct)

    splits = {
        "train": img_files[:train_cutoff],
        "val": img_files[train_cutoff:val_cutoff],
        "test": img_files[val_cutoff:],
    }

    counters = {k: 0 for k in splits}

    for split_name, files in splits.items():
        dest_img_dir = {
            "train": train_img_path,
            "val": val_img_path,
            "test": test_img_path,
        }[split_name]
        dest_lbl_dir = {
            "train": train_lbl_path,
            "val": val_lbl_path,
            "test": test_lbl_path,
        }[split_name]

        for img_path in files:
            # Copy image
            shutil.copy2(img_path, dest_img_dir / img_path.name)

            # Copy label if it exists
            label_path = input_label_path / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, dest_lbl_dir / label_path.name)
            counters[split_name] += 1

    # Summary
    print("[DONE] Split complete:")
    print(f"   Train images: {counters['train']}")
    print(f"   Val   images: {counters['val']}")
    print(f"   Test  images: {counters['test']}")
    print(f"Output saved to: {base_out}\n")


if __name__ == "__main__":
    main()
