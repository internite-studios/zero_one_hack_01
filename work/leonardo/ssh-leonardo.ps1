# =============================================================================
# SSH into Leonardo — one-command login
# =============================================================================
# Prerequisites:
#   1. .env file exists with your LEONARDO_USER and LEONARDO_PASSWORD
#   2. SSH key set up on Leonardo (or password prompt will appear)
#
# Usage:
#   .\work\leonardo\ssh-leonardo.ps1                        # Default login node
#   .\work\leonardo\ssh-leonardo.ps1 login02                 # Specific login node
#   .\work\leonardo\ssh-leonardo.ps1 -interactive            # Interactive GPU session (1 GPU, 2h)
#   .\work\leonardo\ssh-leonardo.ps1 -interactive -InteractiveGpus 4 -InteractiveTime 04:00:00
#   .\work\leonardo\ssh-leonardo.ps1 -setup                  # First-time setup on Leonardo
#   .\work\leonardo\ssh-leonardo.ps1 -sync                   # Rsync local repo to Leonardo
#   .\work\leonardo\ssh-leonardo.ps1 -Command "squeue --me"  # Run single command
# =============================================================================

param(
    [string]$Node = "",
    [switch]$Interactive,
    [string]$InteractiveTime = "02:00:00",
    [int]$InteractiveGpus = 1,
    [switch]$Setup,
    [switch]$Sync,
    [string]$Command = ""
)

# =============================================================================
# Load .env
# =============================================================================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ScriptDir ".env"
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.+)\s*$') {
            $key = $Matches[1].Trim()
            $value = $Matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}
else {
    Write-Host "ERROR: .env file not found at $EnvFile" -ForegroundColor Red
    Write-Host "Run: cp .env.example .env  and fill in your credentials" -ForegroundColor Yellow
    exit 1
}

$User = $env:LEONARDO_USER
$Password = $env:LEONARDO_PASSWORD
$Hostname = if ($Node) { "$Node-ext.leonardo.cineca.it" } else { $env:LEONARDO_HOST }
$Reservation = $env:SLURM_RESERVATION

if (-not $User) {
    Write-Host "ERROR: LEONARDO_USER not set in .env" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Helper: Build ssh command with optional password
# =============================================================================
function Invoke-LeonardoSSH {
    param(
        [string]$TargetHost,
        [string]$RemoteCommand = "",
        [switch]$ForceTty
    )

    # Use sshpass if password is available, otherwise plain ssh
    $sshArgs = @()
    if ($Password) {
        # sshpass handles the password automatically
        $sshCmd = "sshpass -p `"$Password`" ssh -o StrictHostKeyChecking=accept-new"
    } else {
        $sshCmd = "ssh -o StrictHostKeyChecking=accept-new"
    }

    if ($ForceTty) {
        $sshCmd += " -t"
    }

    if ($RemoteCommand) {
        $fullCmd = "$sshCmd ${User}@${TargetHost} `"$RemoteCommand`""
    } else {
        $fullCmd = "$sshCmd ${User}@${TargetHost}"
    }

    Write-Host "Connecting to $User@$TargetHost ..." -ForegroundColor Cyan
    Invoke-Expression $fullCmd
}

# =============================================================================
# Sync mode — rsync local repo to Leonardo
# =============================================================================
if ($Sync) {
    $RemoteDir = $env:LEONARDO_PROJECT_DIR
    if (-not $RemoteDir) {
        $RemoteDir = "/scratch/$User/zero-one-hack"
    }

    Write-Host "Syncing $RepoRoot -> ${User}@${Hostname}:${RemoteDir}" -ForegroundColor Cyan
    Write-Host ""

    # Ensure remote directory exists
    if ($Password) {
        sshpass -p "$Password" ssh -o StrictHostKeyChecking=accept-new "${User}@${Hostname}" "mkdir -p $RemoteDir"
    } else {
        ssh -o StrictHostKeyChecking=accept-new "${User}@${Hostname}" "mkdir -p $RemoteDir"
    }

    # Rsync (exclude git, .pixi, caches, large artifacts, secrets)
    if ($Password) {
        $env:RSYNC_PASSWORD = $Password
        sshpass -p "$Password" rsync -avz --progress `
            --exclude '.git' `
            --exclude '.pixi' `
            --exclude '__pycache__' `
            --exclude '*.pyc' `
            --exclude 'checkpoints' `
            --exclude 'logs' `
            --exclude 'wandb' `
            --exclude '.env' `
            "$RepoRoot/" "${User}@${Hostname}:${RemoteDir}/"
    } else {
        rsync -avz --progress `
            --exclude '.git' `
            --exclude '.pixi' `
            --exclude '__pycache__' `
            --exclude '*.pyc' `
            --exclude 'checkpoints' `
            --exclude 'logs' `
            --exclude 'wandb' `
            --exclude '.env' `
            "$RepoRoot/" "${User}@${Hostname}:${RemoteDir}/"
    }

    Write-Host ""
    Write-Host "Sync complete." -ForegroundColor Green
    exit 0
}

# =============================================================================
# Setup mode — run setup-leonardo.sh on remote
# =============================================================================
if ($Setup) {
    Write-Host "Running first-time setup on Leonardo as $User..." -ForegroundColor Cyan
    Write-Host ""

    $SetupScript = Join-Path $ScriptDir "setup-leonardo.sh"

    if ($Password) {
        sshpass -p "$Password" ssh -o StrictHostKeyChecking=accept-new "${User}@${Hostname}" "bash -s" < $SetupScript
    } else {
        ssh -o StrictHostKeyChecking=accept-new "${User}@${Hostname}" "bash -s" < $SetupScript
    }

    Write-Host ""
    Write-Host "Setup complete. Now run: .\ssh-leonardo.ps1 -sync" -ForegroundColor Green
    exit 0
}

# =============================================================================
# Command mode — run single command and exit
# =============================================================================
if ($Command) {
    $RemoteDir = $env:LEONARDO_PROJECT_DIR
    if (-not $RemoteDir) {
        $RemoteDir = "/scratch/$User/zero-one-hack"
    }
    Invoke-LeonardoSSH -TargetHost $Hostname -RemoteCommand "cd $RemoteDir && $Command"
    exit 0
}

# =============================================================================
# Interactive GPU session mode
# =============================================================================
if ($Interactive) {
    $RemoteDir = $env:LEONARDO_PROJECT_DIR
    if (-not $RemoteDir) {
        $RemoteDir = "/scratch/$User/zero-one-hack"
    }

    $Mem = $InteractiveGpus * 120
    $Cpus = $InteractiveGpus * 8

    $SrunCmd = "srun --partition=boost_usr_prod --reservation=$Reservation --nodes=1 --ntasks=1 --cpus-per-task=$Cpus --gpus-per-task=$InteractiveGpus --mem=${Mem}GB --time=$InteractiveTime --pty /bin/bash"

    Write-Host "Requesting interactive GPU session..." -ForegroundColor Cyan
    Write-Host "  GPUs: $InteractiveGpus | CPUs: $Cpus | Memory: ${Mem}GB | Time: $InteractiveTime" -ForegroundColor Cyan
    Write-Host "  Reservation: $Reservation" -ForegroundColor Cyan
    Write-Host ""

    Invoke-LeonardoSSH -TargetHost $Hostname -RemoteCommand "cd $RemoteDir && $SrunCmd" -ForceTty
    exit 0
}

# =============================================================================
# Default: interactive SSH login
# =============================================================================
Invoke-LeonardoSSH -TargetHost $Hostname
