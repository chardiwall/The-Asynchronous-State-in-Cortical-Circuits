#!/bin/bash
# Fig. 2C's current-correlation components (c_EE, c_II, c_EI): same 347-task
# grid as block2_sweep.slurm, ~6% more compute per task (subsampled current
# recording, see fast_model._run_jit_current). Memory here is much more
# uniform across N than block2_sweep.slurm's: it only stores a fixed-size (up to
# fig2g_subsample_neurons) subsample of continuous current, not the full
# population's binary state, so a single --mem=6G (submitted 2026-09-12,
# worst case ~4.6GB at N=8192) covers the whole grid at near-20-way
# concurrency -- no per-tier submission needed here, unlike block2_sweep.slurm.
#   sbatch --mem=6G slurm/block2_sweep_current.slurm
#SBATCH --job-name=block2_full_pass_current
#SBATCH --partition=gpu
#SBATCH --nodelist=spark-c3d8
#SBATCH --array=0-346%20
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --output=artifacts/block2_full_pass_current/slurm_%a.out

cd ~/programs/The-Asynchronous-State-in-Cortical-Circuits
PYTHONPATH=src .env/bin/python -m block2.run current "$SLURM_ARRAY_TASK_ID"
