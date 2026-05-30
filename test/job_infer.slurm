#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:10:00
#SBATCH --job-name=mnist-infer
#SBATCH --output=test/extras/mnist_infer_%j.out
#SBATCH --error=test/extras/mnist_infer_%j.err

# ---------------------------------------------------------------------------
# SLURM Job: MNIST CNN Inference
# ---------------------------------------------------------------------------
# Runs inference on 10 test digits, saves PNGs and confidence scores.
# Submit AFTER train.py has completed and mnist_cnn.pth exists:
#   sbatch test/job_infer.slurm
# ---------------------------------------------------------------------------

echo "============================================"
echo "  MNIST CNN Inference Job"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $SLURM_NODELIST"
echo "  GPUs:     $SLURM_GPUS_PER_TASK"
echo "  Started:  $(date)"
echo "============================================"

# Point pixi cache to SCRATCH
export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache

# Quick GPU status
echo ""
echo "[GPU Status]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not available"

# Check model exists
echo ""
echo "[Check]"
if [ -f "test/mnist_cnn.pth" ]; then
    echo "  Model file found: test/mnist_cnn.pth ($(du -h test/mnist_cnn.pth | cut -f1))"
else
    echo "  ERROR: test/mnist_cnn.pth not found! Run train job first."
    exit 1
fi

# Run inference
echo ""
echo "[Inference]"
echo "-------------------------------------------"

export RUN_COMMAND="pixi run --manifest-path test/pixi.toml --environment gpu --as-is"
$RUN_COMMAND python3 test/infer.py

echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "  Exit code:    $?"
echo ""
echo "  Results:"
echo "    Input PNGs:  test/input/"
echo "    Confidences: test/output/confidences.json"
echo "    Summary CSV: test/output/summary.csv"
echo "============================================"
