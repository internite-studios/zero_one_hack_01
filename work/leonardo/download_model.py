#!/usr/bin/env python3
"""
Download an open-source LLM from HuggingFace to $SCRATCH/models/.

Run on Leonardo login node (has internet):
    pixi run -e default python work/leonardo/download_model.py

Or with a specific model:
    pixi run -e default python work/leonardo/download_model.py --model Qwen/Qwen2.5-7B-Instruct

The model is saved to $SCRATCH/models/<model-name>/
HF cache is redirected to $SCRATCH/.cache/huggingface/ to avoid filling $HOME.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download an LLM from HuggingFace")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory to store models (default: $SCRATCH/models)",
    )
    args = parser.parse_args()

    # --- Determine storage paths ---
    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'unknown')}")
    models_dir = Path(args.models_dir or f"{scratch}/models")
    cache_dir = Path(os.environ.get("HF_HOME", f"{scratch}/.cache/huggingface"))

    model_slug = args.model.replace("/", "--")
    model_path = models_dir / model_slug

    print("=" * 60)
    print(f" Model Download: {args.model}")
    print(f" Target:          {model_path}")
    print(f" Cache:           {cache_dir}")
    print(f" Scratch:         {scratch}")
    print("=" * 60)
    print()

    # --- Ensure directories exist ---
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Redirect HF cache to scratch
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")

    # --- Check if model already downloaded ---
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Model already exists at {model_path}")
        print("Skipping download. Delete this directory to re-download.")
        print()
        _print_model_files(model_path)
        return

    # --- Download ---
    print(f"Downloading {args.model} ...")
    print("(This may take several minutes depending on model size)")
    print()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pixi install -e default")
        sys.exit(1)

    # Download tokenizer first (small)
    print("[1/2] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        cache_dir=str(cache_dir / "transformers"),
    )

    # Download model (large)
    print("[2/2] Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="auto",
        device_map="cpu",  # Download to CPU only (no GPU on login node)
        trust_remote_code=True,
        cache_dir=str(cache_dir / "transformers"),
    )

    # --- Copy from HF cache to our models directory for easy reference ---
    print()
    print(f"Saving to {model_path} ...")
    tokenizer.save_pretrained(str(model_path))
    model.save_pretrained(str(model_path))

    print()
    print("=" * 60)
    print(" Download complete!")
    print(f" Model:  {model_path}")
    _print_model_files(model_path)
    print()
    print("Next: Submit the inference job:")
    print("  sbatch work/leonardo/job_inference.slurm")
    print("=" * 60)


def _print_model_files(path: Path):
    """Print size summary of downloaded model files."""
    total_size = 0
    file_count = 0
    for f in sorted(path.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            total_size += size_mb
            file_count += 1
            # Only print the key files, not all shards
            if f.suffix in (".json", ".txt", ".md", ".model") or file_count <= 5:
                print(f"  {f.name:50s} {size_mb:8.1f} MB")

    if file_count > 5:
        print(f"  ... and {file_count - 5} more file(s)")

    print(f"  {'TOTAL':50s} {total_size:8.1f} MB")


if __name__ == "__main__":
    main()
