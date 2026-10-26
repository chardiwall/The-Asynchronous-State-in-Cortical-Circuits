#!/bin/bash
# Fig. 3A and 3B: ONE network, one run, both panels (block3.figures_ab writes the raster,
# the 3 ms tracking curves and the measured/jittered correlation histograms).
#
# Not an array job: 3A is one illustrative network and 3B's histogram is over pairs within
# that network, not over networks. Length comes from spiking_network.panels.length_s.
#
# 5000 s, the length Fig. S6 states. Since block 3 integrates its synapses per neuron, a
# 2000 s run measured 57 min wall and 33 GB peak RSS, so 5000 s is ~2.5 h. Memory is the
# cost: the Fig. 3B rate matrix is 40 GB and numpy.corrcoef adds a centred copy, hence
# --mem=100G. block3.figures_ab refuses to start if the matrix would exceed
# panels.max_rate_matrix_gb, so a misconfiguration fails in seconds, not after the run.
#   sbatch slurm/block3_panels.slurm
#SBATCH --job-name=block3_panels
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/block3_panels/slurm.out
#SBATCH --container-image=/home/chardiwall/.local/share/asyncstate-py312.sqsh
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
# The image is ubuntu:24.04 plus python3.12-dev/build-essential, saved once with
#   srun --container-image=ubuntu:24.04 --container-save=<that path> ... apt-get install ...
# so array tasks never pull from Docker Hub (~100 tasks at once hit its HTTP 429 limit).
# With a plain ubuntu image instead, the install below runs per task.
[ -f /usr/include/python3.12/Python.h ] || { apt-get update -qq && apt-get install -y -qq python3.12-dev python3 build-essential time >/dev/null 2>&1; }

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits
mkdir -p artifacts/block3_panels
/usr/bin/time -v -o "artifacts/block3_panels/time.log" \
    env PYTHONPATH=src .env/bin/python -m block3.run panels
