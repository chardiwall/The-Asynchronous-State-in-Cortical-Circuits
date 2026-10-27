#!/bin/bash
# LSM asynchrony test: one array task per (variant, N, network) background run, the grid of
# lsm.run.asynchrony_grid (30 tasks at config.yaml's defaults; 21-29 are lsm.designs). 100 s per run; the N=5000
# faithful tasks are block 3 itself and take ~15 min, the N=500 ones a few minutes.
#SBATCH --job-name=lsm_asynchrony
#SBATCH --partition=gpu
#SBATCH --array=0-29
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/lsm/asynchrony/slurm_%a.out
#SBATCH --container-image=/home/chardiwall/.local/share/asyncstate-py312.sqsh
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
# The image is ubuntu:24.04 plus python3.12-dev/build-essential, saved once with
#   srun --container-image=ubuntu:24.04 --container-save=<that path> ... apt-get install ...
# so array tasks never pull from Docker Hub (~100 tasks at once hit its HTTP 429 limit).
# With a plain ubuntu image instead, the install below runs per task.
[ -f /usr/include/python3.12/Python.h ] || { apt-get update -qq && apt-get install -y -qq python3.12-dev python3 build-essential >/dev/null 2>&1; }

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits
PYTHONPATH=src .env/bin/python -m lsm.run asynchrony "$SLURM_ARRAY_TASK_ID"
