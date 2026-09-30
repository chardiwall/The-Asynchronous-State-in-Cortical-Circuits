"""Block 2's full pass (Fig. 2C's N-grid, config.yaml's binary_network.full_pass):
one independent OS process per (N, realisation) task, dispatched by a Slurm array
(not Python multiprocessing -- docs/adr/0003's ARM64 ProcessPoolExecutor crash).
Each task runs ONE realisation at production length_tau and writes its own summary
row, so parallel tasks never contend on a file and no raw per-neuron trace is kept
for any of the (up to 50) repeats per size.

This module also holds the task-dispatch and aggregation seam full_pass_current.py
reuses -- the two passes differ only in what each task records.

Usage (one task): python -m block2.full_pass <task_index>
Usage (after the array finishes): python -m block2.full_pass --aggregate
"""
import csv
import glob
import json
import os
import sys

from block2.connectivity import couplings
from block2.eval import population_averaged_correlation
from block2.fast_model import simulate_fast_one
from config import load_config

RESULTS_DIR = "artifacts/block2_full_pass"
FIELDS = ["n", "realisation", "rate_E", "rate_I", "rate_X", "r_EE", "r_II", "r_EI", "r_EX", "r_IX"]


def build_full_pass_grid(sizes: list[int], repeats: list[int]) -> list[dict]:
    return [{"n": n, "realisation": r} for n, reps in zip(sizes, repeats) for r in range(reps)]


def task_parameters(task_index: int) -> tuple[dict, dict, dict, dict]:
    """(config, binary_network config, j couplings, this task's {n, realisation}) for one
    Slurm array index. Both full passes walk the same grid, so they dispatch identically.
    """
    config = load_config("config.yaml")
    net = config["binary_network"]
    grid = build_full_pass_grid(net["full_pass"]["sizes"], net["full_pass"]["repeats"])
    return config, net, couplings(net), grid[task_index]


def write_task(results_dir: str, task_index: int, row: dict) -> None:
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/task_{task_index:04d}.json", "w") as f:
        json.dump(row, f)
    print(f"task {task_index} done: {row}", flush=True)


def aggregate(results_dir: str, out_csv: str, fields: list[str]) -> int:
    """Combines every per-task JSON row into one CSV sorted by (n, realisation). Run
    after the whole Slurm array has finished.
    """
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/task_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["n"], r["realisation"]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def run_one_task(
    n: int, realisation: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int,
) -> dict:
    result = simulate_fast_one(n=n, p=p, j=j, m_x=m_x, theta=theta, length_tau=length_tau,
                                sampling_rate=sampling_rate, burn_in_tau=burn_in_tau,
                                seed=seed + realisation)
    e, i, x = result[:n], result[n:2 * n], result[2 * n:]  # views, not copies
    return {
        "n": n, "realisation": realisation,
        "rate_E": float(e.mean()), "rate_I": float(i.mean()), "rate_X": float(x.mean()),
        "r_EE": population_averaged_correlation(e, e, True),
        "r_II": population_averaged_correlation(i, i, True),
        "r_EI": population_averaged_correlation(e, i, False),
        "r_EX": population_averaged_correlation(e, x, False),
        "r_IX": population_averaged_correlation(i, x, False),
    }


def main():
    out_csv = "artifacts/block2_full_pass.csv"
    if sys.argv[1] == "--aggregate":
        print(f"wrote {aggregate(RESULTS_DIR, out_csv, FIELDS)} rows to {out_csv}")
        return

    task_index = int(sys.argv[1])
    config, net, j, task = task_parameters(task_index)
    row = run_one_task(
        n=task["n"], realisation=task["realisation"], p=net["connection_probability"], j=j,
        m_x=net["m_x"], theta=net["theta"], length_tau=net["full_pass"]["length_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"],
    )
    write_task(RESULTS_DIR, task_index, row)


if __name__ == "__main__":
    main()
