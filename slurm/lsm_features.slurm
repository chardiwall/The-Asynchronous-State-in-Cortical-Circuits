#!/bin/bash
# Reservoir features for one (dataset, variant): one array task per chunk of samples.
#   n=$(PYTHONPATH=src .env/bin/python -m lsm.run nchunks nmnist)
#   sbatch --array=0-$((n-1)) slurm/lsm_features.slurm nmnist faithful
# variant "input" is the no-reservoir baseline (no simulation, seconds per chunk).
# Optional third argument: reservoir size N (default lsm.size), written to <variant>_N<N>.
#SBATCH --job-name=lsm_features
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/lsm/features/slurm_%x_%A_%a.out
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
PYTHONPATH=src .env/bin/python -m lsm.run features "$1" "$2" "$SLURM_ARRAY_TASK_ID" ${3:+--size $3}
