#!/bin/bash
# Fig. 3A and 3B: ONE network, one run, both panels (block3.figures_ab writes the raster,
# the 3 ms tracking curves and the measured/jittered correlation histograms).
#
# Not an array job: 3A is one illustrative network and 3B's histogram is over pairs within
# that network, not over networks. Length comes from spiking_network.panels.length_s.
#
# 1000 s, NOT the 5000 s Fig. S6 states -- a deliberate, documented deviation (GitHub #8).
# Measured 27.1 h per 200 s on this node, so 1000 s is ~135 h, inside the 168 h limit below
# with ~24% margin; 5000 s would be ~28 days uncheckpointed. The cost is that Fig. 3B's
# jittered null is 0.0056 wide rather than 0.0027, so the separation from the measured
# histogram is real but weaker than the published panel's. config.yaml has the numbers.
#
# --mem=24G: the rate matrix is 8GB at this length and numpy.corrcoef adds a centred copy,
# on top of the ~4GB network state measured for the 200 s run. block3.figures_ab refuses to start
# if the matrix would exceed panels.max_rate_matrix_gb, so a misconfiguration fails in
# seconds rather than after five days.
#   sbatch slurm/block3_panels.slurm
#SBATCH --job-name=block3_panels
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=14:00:00
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
