#!/bin/bash
# Fig. 3C-D: one array task per (network, current condition). With config.yaml's eight
# I_app levels and ten networks that is 10 * (8 + 1) = 90 tasks -- the "+1" is the
# EPSP-IPSP cross condition Fig. 3C's gold curve needs (block3.vm_ccg.build_task_grid).
#
# Each task runs 50 s of simulated time (SOM S-p.21) on the full 9M-synapse network.
# Measured reference from the Fig. 3A-B full pass on this node: 200 s cost 27h07m wall
# and 3.85GB peak RSS, so 50 s extrapolates to ~7h. --time is set well above that because
# this node's throughput is sensitive to concurrent load (see PROGRESS.md 2026-09-21), and
# --mem is set from that measured 3.85GB plus the recorded-voltage monitor: 20 cells x
# (50s / 1ms) samples x 8 bytes is only 8MB, so the network state still dominates.
#
# Container wrapper: spark-c3d8 has no python3.12-dev and no sudo (docs/adr/0005).
#   sbatch src/block3/vm_ccg.slurm
#SBATCH --job-name=block3_vm_ccg
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --array=0-89%6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/block3_vm_ccg/slurm_%a.out
#SBATCH --container-image=ubuntu:24.04
#SBATCH --container-remap-root
#SBATCH --container-mount-home
#SBATCH --container-writable

set -e
apt-get update -qq
apt-get install -y -qq python3.12-dev python3 build-essential time >/dev/null 2>&1

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits.v2
mkdir -p artifacts/block3_vm_ccg
/usr/bin/time -v -o "artifacts/block3_vm_ccg/time_${SLURM_ARRAY_TASK_ID}.log" \
    env PYTHONPATH=src .env/bin/python -m block3.vm_ccg "$SLURM_ARRAY_TASK_ID"
