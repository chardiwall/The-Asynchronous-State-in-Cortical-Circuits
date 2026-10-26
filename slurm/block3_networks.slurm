#!/bin/bash
# Block 3's full pass (Fig. 3A-B): one array task per network
# realisation, 10 tasks (config.yaml: spiking_network.simulation.n_networks). Wrapped in a
# pyxis/enroot container to work around spark-c3d8 missing python3.12-dev with no sudo
# access -- every task pays a ~1min apt-get install cost, negligible next
# to the ~67h/network run itself.
#
# --mem=6G is a computed, not guessed, estimate: block3.measure's r_bar-over-1000-neuron-subsample
# step builds a (1000, ~199850) float64 rate matrix, exactly 1.6GB by construction
# (n_neurons * n_windows * 8 bytes) -- the dominant, calculable cost. The remaining ~4.4GB
# margin covers the ~9M-synapse network's own state (g_syn/delay/s/x arrays) and Brian2's
# Cython-codegen overhead, neither precisely computed. `/usr/bin/time -v` wraps the actual
# task to capture real peak RSS from production data (block 2's first --mem estimate for
# its full pass was wrong twice before being measured directly -- don't repeat that here
# without a check).
#SBATCH --job-name=block3_full_pass
#SBATCH --partition=gpu
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH --output=artifacts/block3_full_pass/slurm_%a.out
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
mkdir -p artifacts/block3_full_pass
/usr/bin/time -v -o "artifacts/block3_full_pass/time_${SLURM_ARRAY_TASK_ID}.log" \
    env PYTHONPATH=src .env/bin/python -m block3.run networks "$SLURM_ARRAY_TASK_ID"
