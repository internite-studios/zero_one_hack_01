#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --job-name=mnist-cnn
#SBATCH --output=test/extras/mnist_train_%j.out
#SBATCH --error=test/extras/mnist_train_%j.err

# ---------------------------------------------------------------------------
# SLURM Job: MNIST CNN Training on Leonardo
# ---------------------------------------------------------------------------
# Submit this AFTER running download_data.py on the login node:
#   sbatch test/job_train.slurm
#
# Monitor progress:
#   squeue --me
#   tail -f test/extras/mnist_train_<jobid>.out
# ---------------------------------------------------------------------------

echo "============================================"
echo "  MNIST CNN Training Job"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $SLURM_NODELIST"
echo "  GPUs:     $SLURM_GPUS_PER_TASK"
echo "  Started:  $(date)"
echo "============================================"

# ---- Environment setup ----
# Point pixi cache to SCRATCH so we don't fill $HOME (50 GB quota)
export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache

# ---- GPU check ----
echo ""
echo "[GPU Status]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi not available"
echo "  PyTorch CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"

# ---- Train ----
echo ""
echo "[Training]"
echo "-------------------------------------------"

# Run inside the pixi GPU environment
export RUN_COMMAND="pixi run --manifest-path test/pixi.toml --environment gpu --as-is"
$RUN_COMMAND python3 test/train.py

# ---- Done ----
echo ""
echo "============================================"
echo "  Job finished: $(date)"
echo "  Exit code:    $?"
echo "============================================"
