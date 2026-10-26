#!/bin/bash
# Repeat reliability for one (dataset, variant): the stimulus-locked share of the liquid
# state, from the same test samples presented twice with independent noise.
#   sbatch slurm/lsm_reliability.slurm dvs_gesture faithful
# variant "input" measures the encoding noise alone.
#SBATCH --job-name=lsm_reliability
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/lsm/features/slurm_%x_%j.out
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
PYTHONPATH=src .env/bin/python -m lsm.run reliability "$1" "$2" ${3:+--size $3}
