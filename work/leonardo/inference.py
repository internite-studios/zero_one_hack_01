#!/usr/bin/env python3
"""
Load an LLM from local storage on $SCRATCH and run inference on a GPU node.

Usage (inside pixi environment on a Leonardo GPU node):
    python work/leonardo/inference.py
    python work/leonardo/inference.py --model /scratch/$USER/models/Qwen--Qwen2.5-7B-Instruct
    python work/leonardo/inference.py --prompt "What is quantum computing?"
    python work/leonardo/inference.py --prompt-file work/leonardo/prompt.txt

Output is printed to stdout and also saved to results/inference_output.txt.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def get_default_model_path() -> Path:
    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'unknown')}")
    return Path(scratch) / "models" / "Qwen--Qwen2.5-7B-Instruct"


def get_default_prompt_file() -> Path:
    return Path(__file__).resolve().parent / "prompt.txt"


def main():
    parser = argparse.ArgumentParser(description="Run LLM inference on Leonardo GPU")
    parser.add_argument(
        "--model",
        default=str(get_default_model_path()),
        help="Path to downloaded model directory",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to send to the model (overrides --prompt-file)",
    )
    parser.add_argument(
        "--prompt-file",
        default=str(get_default_prompt_file()),
        help="File containing the prompt (default: prompt.txt)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: results/inference_output.txt)",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    scratch = os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', 'unknown')}")

    # Resolve prompt
    if args.prompt:
        prompt = args.prompt
    elif args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"ERROR: Prompt file not found: {prompt_path}")
            sys.exit(1)
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    else:
        prompt = "Hello! Who are you and what can you do?"

    # Output file
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path.cwd() / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"inference_output_{timestamp}.txt"

    # Redirect HF cache to scratch
    cache_dir = Path(os.environ.get("HF_HOME", f"{scratch}/.cache/huggingface"))
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")

    # =========================================================================
    # Print setup info
    # =========================================================================
    header = f"""
======================================================================
 LLM Inference on Leonardo
======================================================================
Timestamp:   {datetime.now().isoformat()}
Model path:  {model_path}
Prompt:      {prompt[:120]}{'...' if len(prompt) > 120 else ''}
Max tokens:  {args.max_new_tokens}
Temperature: {args.temperature}
Output:      {output_path}
======================================================================
"""
    print(header)

    # =========================================================================
    # Check GPU
    # =========================================================================
    print("[GPU Check]")
    try:
        import torch
        print(f"  PyTorch:    {torch.__version__}")
        print(f"  CUDA build: {torch.version.cuda}")
        print(f"  CUDA avail: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  GPU count:  {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU[{i}]:     {props.name}")
                print(f"  GPU[{i}] VRAM: {props.total_mem / (1024**3):.1f} GB")
        else:
            print("  WARNING: No GPU detected! Running on CPU will be very slow.")
            print("  Make sure you submitted this as a GPU job: sbatch job_inference.slurm")
    except ImportError:
        print("  ERROR: torch not installed. Run: pixi install -e gpu")
        sys.exit(1)

    # =========================================================================
    # Load model
    # =========================================================================
    print()
    print("[Loading Model]")

    if not model_path.exists():
        print(f"  ERROR: Model not found at {model_path}")
        print(f"  Run download first: pixi run -e default python work/leonardo/download_model.py")
        sys.exit(1)

    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )

    print(f"  Loading model weights from {model_path} ...")
    print(f"  (This may take 30-60 seconds for a 7B model)")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")

    # =========================================================================
    # Run inference
    # =========================================================================
    print()
    print("[Inference]")
    print(f"  Prompt: {prompt}")
    print()

    # Format as chat if the tokenizer supports it
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"  Input tokens: {inputs.shape[1]}")
    print(f"  Generating up to {args.max_new_tokens} tokens ...")
    print()

    t1 = time.time()

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_time = time.time() - t1
    new_tokens = outputs.shape[1] - inputs.shape[1]
    tokens_per_sec = new_tokens / gen_time if gen_time > 0 else 0

    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # For chat-formatted input, extract just the assistant's response
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # The response includes the chat template markers — try to extract
        # the assistant part
        if "assistant" in response.lower():
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Remove trailing special tokens
                if response.startswith("\n"):
                    response = response[1:]

    # =========================================================================
    # Print results
    # =========================================================================
    separator = "-" * 60
    result = f"""
{separator}
 RESPONSE
{separator}
{response}
{separator}
 STATS
{separator}
  Load time:     {load_time:.1f}s
  Generation:    {gen_time:.1f}s
  New tokens:    {new_tokens}
  Tokens/sec:    {tokens_per_sec:.1f}
  GPU memory:    {torch.cuda.max_memory_reserved(0) / (1024**3):.1f} GB peak
{separator}
"""

    print(result)

    # =========================================================================
    # Save to file
    # =========================================================================
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(result)

    print(f"Output saved to: {output_path}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
