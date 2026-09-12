"""Fig. 2C's current-correlation components (S-Eq 32-33): c_EE, c_II, c_EI and
their sum c = c_EE+c_II+2c_EI (M-Eq 3). Same Slurm-array-of-independent-tasks
shape as block2.full_pass (one (N, realisation) job per task), using
fast_model.simulate_fast_current's subsampled current traces instead of the
full-population binary state full_pass.py records.

Usage (one task): python -m block2.full_pass_current <task_index>
"""
import csv
import json
import sys

from block2.eval import population_averaged_correlation
from block2.fast_model import simulate_fast_current
from block2.full_pass import build_full_pass_grid
from config import load_config

RESULTS_DIR = "artifacts/block2_full_pass_current"


def run_one_current_task(
    n: int, realisation: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int, subsample_size: int,
) -> dict:
    current = simulate_fast_current(n=n, p=p, j=j, m_x=m_x, theta=theta, length_tau=length_tau,
                                     sampling_rate=sampling_rate, burn_in_tau=burn_in_tau,
                                     seed=seed + realisation, subsample_size=subsample_size)
    c_ee = population_averaged_correlation(current["E"], current["E"], True)
    c_ii = population_averaged_correlation(current["I"], current["I"], True)
    c_ei = population_averaged_correlation(current["E"], current["I"], False)
    return {"n": n, "realisation": realisation, "c_EE": c_ee, "c_II": c_ii, "c_EI": c_ei,
            "c_total": c_ee + c_ii + 2 * c_ei}


def aggregate(results_dir: str = RESULTS_DIR,
              out_csv: str = "artifacts/block2_full_pass_current.csv") -> int:
    import glob
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/task_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["n"], r["realisation"]))

    fields = ["n", "realisation", "c_EE", "c_II", "c_EI", "c_total"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    if sys.argv[1] == "--aggregate":
        n = aggregate()
        print(f"wrote {n} rows to artifacts/block2_full_pass_current.csv")
        return

    task_index = int(sys.argv[1])
    config = load_config("config.yaml")
    net = config["binary_network"]
    full_pass = net["full_pass"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}

    grid = build_full_pass_grid(full_pass["sizes"], full_pass["repeats"])
    task = grid[task_index]

    row = run_one_current_task(
        n=task["n"], realisation=task["realisation"], p=net["connection_probability"], j=j,
        m_x=net["m_x"], theta=net["theta"], length_tau=full_pass["length_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"], subsample_size=net["fig2g_subsample_neurons"],
    )
    with open(f"{RESULTS_DIR}/task_{task_index:04d}.json", "w") as f:
        json.dump(row, f)
    print(f"task {task_index} (N={task['n']}, realisation={task['realisation']}) done: {row}", flush=True)


if __name__ == "__main__":
    main()
