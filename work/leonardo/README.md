# work/leonardo — Leonardo Supercomputer Toolkit

Everything you need to get code running on the CINECA Leonardo GPU cluster (NVIDIA A100s, 64GB each, 4 per node).

## Prerequisites (on your local Windows machine)

Install **sshpass** so the PowerShell script can log you in without typing your password:

```powershell
# Option A: via Chocolatey
choco install sshpass

# Option B: via Git Bash
# In Git Bash terminal: pacman -S sshpass

# Option C: manual download
# https://sourceforge.net/projects/sshpass/
```

---

## Quick Start: Run an LLM on Leonardo

This is the end-to-end flow — from zero to an LLM answering your question on an A100.

### 1. One-time setup

```powershell
.\work\leonardo\ssh-leonardo.ps1 -setup
```

Installs pixi, creates project directories on `$SCRATCH`, generates SSH key, tests environment.

### 2. Sync your code to Leonardo

```powershell
.\work\leonardo\ssh-leonardo.ps1 -sync
```

### 3. Download the model (on login node — has internet)

```powershell
# SSH into Leonardo
.\work\leonardo\ssh-leonardo.ps1

# Once on Leonardo, install the pixi environment and download the model:
cd ~/zero-one-hack
pixi install -e default
pixi run -e default python work/leonardo/download_model.py
```

This downloads `Qwen/Qwen2.5-7B-Instruct` (~15GB) to `$SCRATCH/models/`.  
To use a different model: `pixi run -e default python work/leonardo/download_model.py --model mistralai/Mistral-7B-Instruct-v0.3`

### 4. Run inference (on GPU compute node)

```bash
sbatch work/leonardo/job_inference.slurm
```

This submits a 1-GPU SLURM job that:
- Loads the model from `$SCRATCH/models/`
- Sends the prompt from `work/leonardo/prompt.txt`
- Prints the response + timing stats
- Saves output to `results/inference_output_<timestamp>.txt`

### 5. Check the result

```bash
# Watch the job progress
squeue --me
tail -f logs/inference_<jobid>.out

# When done, read the result
cat results/inference_output_*.txt
```

### Custom prompts

```bash
# Command-line prompt
sbatch work/leonardo/job_inference.slurm --prompt "Explain quantum computing in 3 sentences"

# Custom model path
sbatch work/leonardo/job_inference.slurm --model /scratch/$USER/models/Mistral-7B-Instruct-v0.3

# Edit prompt.txt and re-run
echo "What is the meaning of life?" > work/leonardo/prompt.txt
sbatch work/leonardo/job_inference.slurm
```

---

## Files

| File | Purpose |
|---|---|
| `.env` | Your credentials (never committed) |
| `.env.example` | Template for new team members |
| `pixi.toml` | pixi manifest — Python 3.12, PyTorch + CUDA 12.4, HuggingFace |
| `ssh-leonardo.ps1` | PowerShell: SSH, sync, setup, interactive sessions, one-off commands |
| `setup-leonardo.sh` | First-time Leonardo setup (pixi, SSH keys, git, directories) |
| `download_model.py` | Downloads an LLM from HuggingFace to `$SCRATCH/models/` |
| `inference.py` | Loads a local model, runs inference on GPU, prints stats + response |
| `prompt.txt` | Default prompt for the LLM (edit this to change the question) |
| `job_1gpu.slurm` | SLURM job — 1 GPU, 120GB RAM, 8 CPUs (general purpose) |
| `job_2gpu.slurm` | SLURM job — 2 GPUs, 240GB RAM, 16 CPUs |
| `job_4gpu.slurm` | SLURM job — 4 GPUs, 480GB RAM, 32 CPUs |
| `job_inference.slurm` | SLURM job — 1 GPU, runs inference.py with the prompt |

---

## Storage on Leonardo

| Storage | Path | Quota | Use for |
|---|---|---|---|
| `$HOME` | `/users/a08trc28` | 50 GB | Config, SSH keys, pixi binary |
| `$SCRATCH` | `/scratch/a08trc28` | No quota | **Everything** during hackathon (auto-deleted after 40 days) |
| `$PUBLIC` | `/public/a08trc28` | 50 GB | Sharing files between users |

`$WORK` and `$FAST` are **not available** during the hackathon — use `$SCRATCH`.

---

## Architecture: How Leonardo Works

```
Your Windows PC                 Leonardo Login Node           Leonardo GPU Node
(has internet)                  (has internet)                (NO internet*)
                                10-min CPU limit              A100 GPU, SLURM job

  1. .\ssh-leonardo.ps1  ──SSH──>  login01-ext
  2. .\ssh-leonardo.ps1 -sync ──>  $SCRATCH/zero-one-hack/
  3. (manual SSH)       ───────>  download_model.py  ──HF──>  $SCRATCH/models/
  4.                     sbatch ─────────────────────────────>  job_inference.slurm
                                        $SCRATCH is shared ──>  inference.py → GPU
  5.                     <────────────────────────────────────  results/inference_output.txt
```

\* Compute nodes can reach the internet only through the HTTP proxy (for small API calls).  
  Large downloads MUST happen on login nodes or via `lrd_all_serial` partition.

---

## ps1 Commands Reference

| Command | What it does |
|---|---|
| `.\ssh-leonardo.ps1` | Interactive SSH (login node) |
| `.\ssh-leonardo.ps1 login05` | Use specific login node |
| `.\ssh-leonardo.ps1 -setup` | First-time setup (install pixi, config git, create dirs) |
| `.\ssh-leonardo.ps1 -sync` | Rsync local repo to `$SCRATCH/zero-one-hack` |
| `.\ssh-leonardo.ps1 -interactive` | Interactive GPU session (1 GPU, 2h) |
| `.\ssh-leonardo.ps1 -interactive -InteractiveGpus 4 -InteractiveTime 04:00:00` | 4 GPU, 4h session |
| `.\ssh-leonardo.ps1 -Command "squeue --me"` | Run single command, print output, exit |

## Handy Leonardo Commands

```bash
squeue --me                                  # Your jobs
scancel <JOBID>                              # Cancel a job
sinfo                                        # Partition/node status
tail -f logs/slurm_<jobid>.out               # Follow job output
tail -f logs/inference_<jobid>.out           # Follow inference output
srun --overlap --pty --jobid=<JOBID> bash    # Shell on running job node

# Long process on login node (login has 10-min CPU limit):
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty bash

# Pull Docker container as Singularity:
srun --partition=lrd_all_serial --time 04:00:00 --gres=tmpfs:100G --mem=16G --pty \
  singularity pull my-container.sif docker://docker.io/some/image:tag

# Run inside container:
singularity exec --nv --bind $SCRATCH:/scratch my-container.sif python3 script.py
```
