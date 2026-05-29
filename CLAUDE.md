# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the central repository for **Zero One Hack_01**, a 36-hour hackathon hosted by [Lumos Consulting](https://lumos-consulting.at) at AI Factory Austria in Vienna. Compute is provided by CINECA on the **Leonardo GPU Cluster** (NVIDIA A100 GPUs).

The hackathon runs three parallel tracks — your team is competing in one of them. Each track has its own briefing, data, and deliverables in `/tracks/`.

## Repository Structure

```
.
├── README.md                  # Hackathon overview, links, rules
├── CLAUDE.md                  # This file
├── submission/
│   ├── SUBMISSION.md          # What/how to submit (Tally form, checklist, deadlines)
│   └── REPORT_TEMPLATE.md     # Template for the required REPORT.md
├── tracks/
│   ├── insurance-uniqa/       # UNIQA health insurance Conversion Coach
│   ├── industrial-infineon/   # Infineon semiconductor process sequences
│   └── forecasting-sybilion/  # Sybilion probabilistic forecasting API
└── extras/                    # (create this) for slides, results, artifacts
```

## The Three Tracks

| Track | Partner | What You Build |
|---|---|---|
| Insurance AI | UNIQA | Conversion Coach that detects abandonment intent in a health insurance calculator and intervenes in real time. Persona-based simulations run on Leonardo. |
| Industrial AI | Infineon | Train and benchmark sequence models on semiconductor process flows. Three eval tasks: next-step prediction, sequence completion, anomaly detection. |
| Forecasting AI | Sybilion | Decision agent on top of a probabilistic forecasting API. Must adapt to a mid-run assumption shift during the Sunday live demo. |

Your chosen track's folder contains the full briefing (EN + DE), all data, and starter utilities. Read the track's README.md first, then the full case spec.

## Leonardo Supercomputer (CINECA)

### SSH Access

For this hackathon, 2FA is **not** used. Connect directly:

```bash
ssh your_username@login01-ext.leonardo.cineca.it
# Alternative login nodes: login02-ext, login05-ext, login07-ext
```

### File Systems

| Variable | Path | Size | Persistence |
|---|---|---|---|
| `$HOME` | `/users/$USER` | 50 GB | Permanent, backed up |
| `$SCRATCH` | `/scratch/$USER` | No quota | **Auto-deleted after 40 days** |
| `$PUBLIC` | `/public/$USER` | 50 GB | Share files between users |

**Important:** `$WORK` and `$FAST` are **NOT available** during the hackathon. Use `$SCRATCH` for all project files, data, and training outputs. Use `$HOME` only for config, SSH keys, and the pixi binary.

### Internet Access on Compute Nodes

**Compute nodes have no direct internet access.** Download all large files (datasets, model weights, packages) from the login nodes before submitting jobs.

For low-bandwidth traffic on compute nodes (e.g., API calls, small downloads), set the proxy:

```bash
export HTTP_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export HTTPS_PROXY=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export http_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
export https_proxy=http://proxyuser:5dd1d2bd00@10.99.0.1:38425
```

**Important:** The proxy restarts periodically (10 min CPU time limit). TCP connections will drop. Only use it for low-bandwidth traffic — never for downloading large models or datasets.

### SLURM Job Submission

Leonardo uses SLURM. Key partition: `boost_usr_prod`. The hackathon reservation is `s_tra_ncc`.

#### 1-GPU Job Script (pixi environment)

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=120GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

export RUN_COMMAND="/path/to/pixi run --as-is"
$RUN_COMMAND python3 script.py
```

#### 2-GPU Job Script (Singularity container)

```bash
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --reservation=s_tra_ncc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=2
#SBATCH --mem=240GB
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00

export CONTAINER="singularity exec --rm container.sif"
$CONTAINER python3 script.py
```

#### Fair-Share Resource Rules

- Memory: 120 GB × number of GPUs
- CPUs: 8 × number of GPUs
- Maximum 4 GPUs per node
- Time limit: up to 24 hours

#### Common SLURM Commands

```bash
sbatch job.sh           # Submit a batch job
squeue --me             # Check your jobs
scancel <JOBID>         # Cancel a job
sinfo                   # Check partition/node status
```

### Interactive Session (for debugging)

```bash
srun --partition=boost_usr_prod --reservation=s_tra_ncc \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --gpus-per-task=1 \
  --mem=120GB --time=02:00:00 --pty /bin/bash
```

## pixi Package Manager

[pixi](https://pixi.sh) is the recommended package manager for this project. It's a fast, Rust-based tool from prefix.dev that combines conda packages with PyPI support, produces deterministic lock files, and handles CUDA/GPU dependencies natively.

### Installation on Leonardo

```bash
curl -fsSL https://pixi.sh/install.sh | sh
# Restart shell or source ~/.bashrc
```

### Project Initialization (run on login node, inside $WORK)

```bash
cd $WORK
mkdir my-project && cd my-project
pixi init

# Add Python + PyTorch with CUDA support
pixi add python=3.12 pytorch torchvision torchaudio

# Add PyPI packages
pixi add --pypi transformers datasets scikit-learn jupyter

# Install everything
pixi install
```

### Running in SLURM Jobs

In your SLURM script, use pixi to execute commands inside the environment:

```bash
# Option A: Run directly
pixi run python train.py

# Option B: Use --as-is to pass shell constructs
pixi run --as-is python train.py --batch-size 64 --epochs 100

# Option C: Specify a manifest path if not in project root
pixi run --manifest-path /path/to/project python train.py
```

### Multi-Environment Setup (CPU dev on laptop, GPU training on Leonardo)

```toml
# pixi.toml skeleton
[project]
name = "zero-one-hack"
channels = ["conda-forge"]
platforms = ["linux-64"]

[dependencies]
python = "3.12.*"
pytorch = ">=2.5"

[feature.dev.dependencies]
pytest = "*"
ruff = "*"
ipython = "*"

[feature.gpu.system-requirements]
cuda = "12"

[feature.gpu.dependencies]
pytorch-cuda = { version = "==12.*", channel = "pytorch" }

[environments]
default = ["dev"]
gpu = ["dev", "gpu"]
```

Use `pixi shell -e gpu` for training, `pixi shell -e default` for local development.

### Running LLM Inference (end-to-end)

A complete pipeline for downloading and running open-source LLMs is in `work/leonardo/`. The scripts:

1. **Download model** on login node (has internet): `pixi run -e default python work/leonardo/download_model.py`
2. **Run inference** on GPU node (via SLURM): `sbatch work/leonardo/job_inference.slurm`
3. **Check results**: `cat results/inference_output_*.txt`

The default model is `Qwen/Qwen2.5-7B-Instruct` (~15GB download, no HF auth needed). Models are stored on `$SCRATCH/models/`. Edit `work/leonardo/prompt.txt` to change the question. See `work/leonardo/README.md` for the full walkthrough.

### Cache on Leonardo

Set `PIXI_CACHE_DIR` to `$CINECA_SCRATCH` to avoid filling your home directory:

```bash
export PIXI_CACHE_DIR=$CINECA_SCRATCH/.pixi-cache
```

## Submission Requirements (Sunday 10:00 deadline)

1. **Tally form** — submitted by 10:00 (submit by 09:45 to be safe)
2. **Public GitHub repo** — MIT licensed, with `README.md`, `REPORT.md`, and `requirements.txt` (or `pixi.toml` + `pixi.lock`)
3. **Slides** — PDF, max 10 slides
4. **Demo video** — MP4, max 2 minutes, 1080p

Full details and checklist in `submission/SUBMISSION.md`. Use `submission/REPORT_TEMPLATE.md` as your starting point for `REPORT.md`.

## Key Constraints & Gotchas

- **No internet on compute nodes.** Download models, datasets, and packages on login nodes before submitting SLURM jobs. The HTTP proxy is for low-bandwidth only and drops connections frequently.
- **$CINECA_SCRATCH auto-purges after 40 days.** Don't store anything permanent there.
- **Track-specific scope boundaries exist.** For insurance: only the private-doctor/"myself"/online-purchasable path is in scope (Start & Optimal tariffs). Hospital path and "other persons" route to advisor — no coaching.
- **No real UNIQA calculator access.** Insurance track demos are simulation-based. Build a journey state machine, persona bots, and coach logic — not a replica of the UNIQA frontend.
- **No basic LLM wrappers.** The jury explicitly rejects thin wrappers around API calls. Your system must have substantive logic of its own.
- **Reproducibility matters.** The jury will try to run your code. A pixi lock file (`pixi.lock`) is the strongest way to guarantee reproducibility.
- **Honest engineering reporting counts.** REPORT.md should document what worked AND what didn't. Mocking/stubbing is fine if disclosed.
- **The eval_metrics.py script** (Infineon track) is provided in `tracks/industrial-infineon/training_data/` — use it for self-evaluation before submitting.
