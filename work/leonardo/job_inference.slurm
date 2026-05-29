#!/bin/bash
# =============================================================================
# Leonardo SLURM Job — LLM Inference (1 GPU)
# =============================================================================
# Runs inference.py using the pixi GPU environment.
#
# Prerequisites:
#   1. Model downloaded to $SCRATCH/models/ (run download_model.py on login node)
#   2. Project synced to $SCRATCH/zero-one-hack/
#
# Usage:
#   sbatch work/leonardo/job_inference.slurm
#   sbatch work/leonardo/job_inference.slurm --prompt "What is machine learning?"
#   sbatch work/leonardo/job_inference.slurm --model /scratch/$USER/models/Qwen--Qwen2.5-7B-Instruct
#
# Arguments after the script name are forwarded to inference.py.
# =============================================================================

#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --job-name=llm-infer
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

# =============================================================================
# Environment
# =============================================================================

# Source .env if present (for proxy, paths)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Point HF cache to scratch (not home — only 50GB)
export HF_HOME="${HF_HOME:-$SCRATCH/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Proxy for any API calls (not needed for local model, but set anyway)
export HTTP_PROXY="${HTTP_PROXY:-http://proxyuser:5dd1d2bd00@10.99.0.1:38425}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxyuser:5dd1d2bd00@10.99.0.1:38425}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# pixi cache on scratch
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$SCRATCH/.pixi-cache}"
mkdir -p "$PIXI_CACHE_DIR"

# Create output directories
mkdir -p logs results

# =============================================================================
# Job Info
# =============================================================================
cat << 'BANNER'
======================================================================
  LLM Inference Job
======================================================================
BANNER

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_JOB_NODELIST"
echo "GPUs:        1"
echo "Time limit:  $SLURM_JOB_TIME"
echo "Project:     $(pwd)"
echo "Model dir:   ${MODEL_DIR:-$SCRATCH/models}"
echo "Arguments:   $@"
echo ""

echo "[GPU Info]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
echo ""

# =============================================================================
# Verify model exists
# =============================================================================
MODEL_DIR="${MODEL_DIR:-$SCRATCH/models}"
DEFAULT_MODEL="$MODEL_DIR/Qwen--Qwen2.5-7B-Instruct"

if [ ! -d "$DEFAULT_MODEL" ] && [ $# -eq 0 ]; then
    echo "============================================================"
    echo " WARNING: Default model not found at:"
    echo "   $DEFAULT_MODEL"
    echo ""
    echo " If you haven't downloaded the model yet, run on login node:"
    echo "   pixi run -e default python work/leonardo/download_model.py"
    echo "============================================================"
    echo ""
fi

# =============================================================================
# Run inference via pixi
# =============================================================================
PIXI="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"

if [ ! -f "$PIXI" ]; then
    echo "ERROR: pixi not found at $PIXI. Run setup-leonardo.sh first."
    exit 1
fi

echo "Installing pixi environment..."
$PIXI install -e gpu

echo ""
echo "Running inference..."
echo "============================================================"
echo ""

# If no arguments passed, use default prompt file
if [ $# -eq 0 ]; then
    $PIXI run -e gpu python work/leonardo/inference.py
else
    $PIXI run -e gpu python work/leonardo/inference.py "$@"
fi

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "Job finished with exit code: $EXIT_CODE"
echo "Output file: results/inference_output_*.txt"
echo "============================================================"

exit $EXIT_CODE
