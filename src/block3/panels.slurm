#!/bin/bash
# Fig. 3A and 3B: ONE network, one run, both panels (block3.panels writes the raster,
# the 3 ms tracking curves and the measured/jittered correlation histograms).
#
# Not an array job: 3A is one illustrative network and 3B's histogram is over pairs within
# that network, not over networks. Length comes from spiking_network.panels.length_s, which
# is 5000 s (Fig. S6's stated single-network protocol) because Fig. 3B's jittered null is
# otherwise as wide as the measurement -- see src/block3/README.md.
#
# 200 s measured 27h07m wall and 3.85GB peak RSS on this node, so 5000 s extrapolates to
# roughly 28 DAYS. That is far past any sensible --time, so this job is checkpoint-free and
# will need either splitting or a faster codegen path before it can actually run; the
# --time below is the cluster maximum, not a sufficient budget. Treat this script as the
# correct specification of the run, not as a submittable job yet.
#   sbatch src/block3/panels.slurm
#SBATCH --job-name=block3_panels
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=168:00:00
#SBATCH --output=artifacts/block3_panels/slurm.out
#SBATCH --container-image=ubuntu:24.04
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
apt-get update -qq
apt-get install -y -qq python3.12-dev python3 build-essential time >/dev/null 2>&1

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits.v2
mkdir -p artifacts/block3_panels
/usr/bin/time -v -o "artifacts/block3_panels/time.log" \
    env PYTHONPATH=src .env/bin/python -m block3.panels
