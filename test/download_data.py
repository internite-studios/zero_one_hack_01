"""
download_data.py — MNIST Dataset Downloader
============================================
!!! IMPORTANT !!!
This script MUST be run on the LOGIN NODE (login01-ext.leonardo.cineca.it).
Leonardo compute nodes have NO internet access — the HTTP proxy is for
low-bandwidth traffic only and drops connections frequently.
DO NOT submit this as a SLURM job.

Usage (run on login node):
    cd /scratch/$USER/zero_one_hack_01
    pixi run --manifest-path test/pixi.toml download
"""

import os
import sys
from pathlib import Path

import torchvision.datasets as datasets

# Download target — store in test/data/ relative to repo root
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  MNIST DATA DOWNLOADER")
print("=" * 60)
print()
print("  REMINDER: You MUST run this on a LOGIN NODE.")
print("  Compute nodes have NO internet — this download WILL FAIL there.")
print()
print(f"  Target directory: {DATA_DIR}")
print()

# Download MNIST training split
print("[1/2] Downloading MNIST training set...")
datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
print("       Done.")

# Download MNIST test split
print("[2/2] Downloading MNIST test set...")
datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
print("       Done.")

# Quick sanity check
expected_files = [
    "MNIST/raw/train-images-idx3-ubyte",
    "MNIST/raw/train-labels-idx1-ubyte",
    "MNIST/raw/t10k-images-idx3-ubyte",
    "MNIST/raw/t10k-labels-idx1-ubyte",
]
all_ok = True
for f in expected_files:
    p = DATA_DIR / f
    if p.exists():
        print(f"  [OK]  {f}")
    else:
        print(f"  [MISSING]  {f}")
        all_ok = False

print()
if all_ok:
    print("  SUCCESS — All MNIST files downloaded.")
    print("  You can now submit the training job:")
    print("      sbatch test/job_train.slurm")
else:
    print("  ERROR — Some files are missing. Try re-running the script.")
    sys.exit(1)
