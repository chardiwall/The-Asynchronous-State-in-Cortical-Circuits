#!/bin/bash
# Block 2's full pass (Fig. 2C's N-grid): one array task per (N, realisation),
# 347 tasks total (config.yaml's binary_network.full_pass), throttled to 20
# concurrent (spark-c3d8's core count). Run from the repo root on the DGX.
#
# --mem MUST be overridden per size tier at submit time, not left at any single
# flat value here: memory need is dominated by the (3N)x(3N)-ish weight
# matrices plus the (3N, n_samples) uint8 activity array, both O(N)-ish, so a
# value sized for N=8192 (~9GB) throttles concurrency to ~11-way for the ENTIRE
# grid (most tasks need only megabytes) -- and a value too small for the
# actual N being run causes real OOM kills, not just queuing (both observed on
# the DGX in session 2026-09-12; see docs/adr/0003). Submit per tier, e.g.:
#   sbatch --array=0-299%20  --mem=1G src/block2/full_pass.slurm   # N<=1780
#   sbatch --array=300-324%20 --mem=3G src/block2/full_pass.slurm  # N=3160
#   sbatch --array=325-336%19 --mem=6G src/block2/full_pass.slurm  # N=5620
#   sbatch --array=337-346%13 --mem=9G src/block2/full_pass.slurm  # N=8192
#SBATCH --job-name=block2_full_pass
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --array=0-346%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=artifacts/block2_full_pass/slurm_%a.out

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits.v2
PYTHONPATH=src .env/bin/python -m block2.full_pass "$SLURM_ARRAY_TASK_ID"
