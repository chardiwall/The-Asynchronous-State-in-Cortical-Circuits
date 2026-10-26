#!/bin/bash
# LSM asynchrony test: one array task per (variant, N, network) background run, the grid of
# lsm.run.asynchrony_grid (21 tasks at config.yaml's defaults). 100 s per run; the N=5000
# faithful tasks are block 3 itself and take ~15 min, the N=500 ones a few minutes.
#SBATCH --job-name=lsm_asynchrony
#SBATCH --partition=gpu
#SBATCH --array=0-20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/lsm/asynchrony/slurm_%a.out
#SBATCH --container-image=ubuntu:24.04
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
apt-get update -qq
apt-get install -y -qq python3.12-dev python3 build-essential >/dev/null 2>&1

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits
PYTHONPATH=src .env/bin/python -m lsm.run asynchrony "$SLURM_ARRAY_TASK_ID"
