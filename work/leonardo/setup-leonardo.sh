#!/bin/bash
# =============================================================================
# Leonardo First-Time Setup Script
# =============================================================================
# Run once after getting access to Leonardo.
# Installs pixi, creates project directories, configures SSH keys and git.
#
# Usage (from your Windows machine):
#   .\work\leonardo\ssh-leonardo.ps1 -setup
#
# Or manually after SSH-ing in:
#   bash work/leonardo/setup-leonardo.sh
# =============================================================================

set -euo pipefail

echo "=============================================="
echo " Leonardo First-Time Setup"
echo " $(date)"
echo "=============================================="
echo ""

# ------------------------------------------------------------------
# 1. Filesystem overview
# ------------------------------------------------------------------
HOME_DIR="$HOME"
SCRATCH_DIR="/scratch/$USER"
PUBLIC_DIR="/public/$USER"

echo "[1/7] Filesystems available:"
echo "  HOME:    $HOME_DIR       (50 GB — config, SSH keys, pixi binary)"
echo "  SCRATCH: $SCRATCH_DIR     (no quota — use for everything during hackathon)"
echo "  PUBLIC:  $PUBLIC_DIR      (50 GB — share files between users)"
echo ""
echo "  NOTE: \$WORK and \$FAST are NOT available during the hackathon."
echo "        Use \$SCRATCH for all project files and data."
echo ""

# ------------------------------------------------------------------
# 2. Create project directories
# ------------------------------------------------------------------
echo "[2/7] Creating project directories..."

PROJECT_DIR="${SCRATCH_DIR}/zero-one-hack"
mkdir -p "$PROJECT_DIR"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "${SCRATCH_DIR}/.pixi-cache"

# Also create a symlink in HOME for convenience
ln -sf "$PROJECT_DIR" "$HOME/zero-one-hack" 2>/dev/null || true

echo "  Project:  $PROJECT_DIR"
echo "  Symlink:  ~/zero-one-hack -> $PROJECT_DIR"
echo "  Cache:    ${SCRATCH_DIR}/.pixi-cache"

# ------------------------------------------------------------------
# 3. Install pixi
# ------------------------------------------------------------------
echo ""
echo "[3/7] Installing pixi..."

if command -v pixi &>/dev/null; then
    echo "  pixi already installed: $(pixi --version)"
else
    echo "  Downloading and installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash

    # Add to current session
    export PATH="$HOME/.pixi/bin:$PATH"

    # Add to .bashrc if not already there
    if ! grep -q '.pixi/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo '' >> "$HOME/.bashrc"
        echo '# pixi package manager' >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> "$HOME/.bashrc"
        echo 'export PIXI_CACHE_DIR="/scratch/$USER/.pixi-cache"' >> "$HOME/.bashrc"
    fi

    echo "  pixi installed: $(pixi --version)"
fi

# ------------------------------------------------------------------
# 4. Check for sshpass / suggest installation
# ------------------------------------------------------------------
echo ""
echo "[4/7] Checking sshpass..."

if command -v sshpass &>/dev/null; then
    echo "  sshpass available (needed on your LOCAL Windows machine, not here)"
else
    echo "  NOTE: sshpass is not installed on Leonardo (and doesn't need to be)."
    echo "        On your LOCAL Windows machine, install sshpass for passwordless SSH:"
    echo "        - Git Bash: pacman -S sshpass"
    echo "        - Or download from: https://sourceforge.net/projects/sshpass/"
fi

# ------------------------------------------------------------------
# 5. Set up SSH key (if not exists)
# ------------------------------------------------------------------
echo ""
echo "[5/7] Checking SSH keys..."

if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo "  Generating new SSH key pair..."
    ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -C "$USER@leonardo"
    echo ""
    echo "  ========================================"
    echo "  Public key (add to GitHub):"
    echo "  ========================================"
    cat "$HOME/.ssh/id_ed25519.pub"
    echo "  ========================================"
    echo ""
    echo "  GitHub: https://github.com/settings/keys"
else
    echo "  SSH key already exists at ~/.ssh/id_ed25519"
fi

# ------------------------------------------------------------------
# 6. Configure git
# ------------------------------------------------------------------
echo ""
echo "[6/7] Configuring git..."

if [ -z "$(git config --global user.name 2>/dev/null)" ]; then
    echo "  NOTE: git user.name not set."
    echo "    Run: git config --global user.name \"Your Name\""
    echo "    Run: git config --global user.email \"your.email@example.com\""
else
    echo "  git user: $(git config --global user.name) <$(git config --global user.email)>"
fi

# ------------------------------------------------------------------
# 7. Test pixi environment
# ------------------------------------------------------------------
echo ""
echo "[7/7] Testing pixi environment..."

cd "$PROJECT_DIR"

if [ -f "pixi.toml" ]; then
    echo "  Running pixi install (this may take a few minutes on first run)..."
    pixi install -e gpu 2>&1 | tail -10
    echo ""
    echo "  Testing GPU availability (login node — expected to show no GPU):"
    pixi run -e gpu python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA build:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  GPU Mem:  {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  NOTE: No GPU visible on login node. This is normal.')
    print('        Submit a SLURM job to access GPUs:')
    print('        sbatch work/leonardo/job_1gpu.slurm python -c \"import torch; print(torch.cuda.get_device_name(0))\"')
" 2>&1 || echo "  (pixi.toml needs to be synced first — run: .\ssh-leonardo.ps1 -sync)"
else
    echo "  No pixi.toml found yet in $PROJECT_DIR."
    echo "  Run this on your local machine to sync:"
    echo "    .\work\leonardo\ssh-leonardo.ps1 -sync"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Setup Complete!"
echo "=============================================="
echo ""
echo "What's next:"
echo "  1. Sync your code:     .\ssh-leonardo.ps1 -sync"
echo "  2. Log in normally:    .\ssh-leonardo.ps1"
echo "  3. Submit a job:       sbatch work/leonardo/job_1gpu.slurm python train.py"
echo "  4. Interactive GPU:    .\ssh-leonardo.ps1 -interactive"
echo "  5. Multi-GPU:          .\ssh-leonardo.ps1 -interactive -InteractiveGpus 4"
echo ""
echo "Useful commands on Leonardo:"
echo "  squeue --me            # Check your jobs"
echo "  scancel <JOBID>        # Cancel a job"
echo "  sinfo                  # Node/partition status"
echo "  tail -f logs/slurm_*.out  # Follow job output"
echo "  srun --overlap --pty --jobid=<JOBID> bash  # Shell on a running job node"
echo ""
