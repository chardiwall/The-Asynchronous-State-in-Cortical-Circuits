#!/bin/bash
# Fig. 2C's current-correlation components (c_EE, c_II, c_EI): same 347-task
# grid as full_pass.slurm, ~6% more compute per task (subsampled current
# recording, see fast_model._run_jit_current). NOT yet submitted -- researcher
# confirmation pending given it needs to queue behind full_pass's job.
#   sbatch src/block2/full_pass_current.slurm
#SBATCH --job-name=block2_full_pass_current
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --array=0-346%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=20:00:00
#SBATCH --output=artifacts/block2_full_pass_current/slurm_%a.out

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits.v2
PYTHONPATH=src .env/bin/python -m block2.full_pass_current "$SLURM_ARRAY_TASK_ID"
