#!/bin/bash
# =============================================================================
# Leonardo SLURM Job — 1 GPU (pixi environment)
# =============================================================================
# Usage:
#   sbatch work/leonardo/job_1gpu.slurm python train.py --epochs 50
#   sbatch work/leonardo/job_1gpu.slurm python eval.py --checkpoint checkpoints/best.pt
#
# All arguments after the script name are forwarded to pixi run -e gpu.
# =============================================================================

#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --job-name=z01-hack
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# =============================================================================
# Environment
# =============================================================================

# Source .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Proxy for low-bandwidth traffic on compute nodes (per official slides)
export HTTP_PROXY="${HTTP_PROXY:-http://proxyuser:5dd1d2bd00@10.99.0.1:38425}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxyuser:5dd1d2bd00@10.99.0.1:38425}"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

# pixi cache on scratch (avoids filling home quota)
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$SCRATCH/.pixi-cache}"
mkdir -p "$PIXI_CACHE_DIR"

# WandB offline mode (compute nodes have no internet)
export WANDB_MODE=offline

# Create log directory on scratch
mkdir -p logs

# =============================================================================
# Job Info
# =============================================================================
echo "=============================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Job Name:    $SLURM_JOB_NAME"
echo "Node:        $SLURM_JOB_NODELIST"
echo "GPUs:        1"
echo "Time limit:  $SLURM_JOB_TIME"
echo "Project dir: $(pwd)"
echo "Arguments:   $@"
echo "=============================================="
echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available (check module load)"
echo ""

# =============================================================================
# Run via pixi
# =============================================================================

PIXI="${PIXI_BIN:-$HOME/.pixi/bin/pixi}"

if [ ! -f "$PIXI" ]; then
    echo "ERROR: pixi not found at $PIXI. Run setup-leonardo.sh first."
    exit 1
fi

echo "Installing pixi environment..."
$PIXI install -e gpu

echo ""
echo "Running: $PIXI run -e gpu $@"
echo "=============================================="
echo ""

$PIXI run -e gpu "$@"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job finished with exit code: $EXIT_CODE"
echo "=============================================="

exit $EXIT_CODE
