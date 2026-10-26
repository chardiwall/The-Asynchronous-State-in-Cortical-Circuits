#!/bin/bash
# Reservoir features for one (dataset, variant): one array task per chunk of samples.
#   n=$(PYTHONPATH=src .env/bin/python -m lsm.run nchunks nmnist)
#   sbatch --array=0-$((n-1)) slurm/lsm_features.slurm nmnist faithful
# variant "input" is the no-reservoir baseline (no simulation, seconds per chunk).
#SBATCH --job-name=lsm_features
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/lsm/features/slurm_%x_%A_%a.out
#SBATCH --container-image=ubuntu:24.04
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
apt-get update -qq
apt-get install -y -qq python3.12-dev python3 build-essential >/dev/null 2>&1

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits
PYTHONPATH=src .env/bin/python -m lsm.run features "$1" "$2" "$SLURM_ARRAY_TASK_ID"
