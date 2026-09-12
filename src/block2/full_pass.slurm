#!/bin/bash
# Block 2's full pass (Fig. 2C's N-grid): one array task per (N, realisation),
# 347 tasks total (config.yaml's binary_network.full_pass), throttled to 20
# concurrent (spark-c3d8's core count). Run from the repo root on the DGX:
#   sbatch src/block2/full_pass.slurm
#SBATCH --job-name=block2_full_pass
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --array=0-346%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --output=artifacts/block2_full_pass/slurm_%a.out

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits.v2
PYTHONPATH=src .env/bin/python -m block2.full_pass "$SLURM_ARRAY_TASK_ID"
