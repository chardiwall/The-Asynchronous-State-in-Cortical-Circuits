"""Block 2's full pass (Fig. 2C's N-grid, config.yaml's binary_network.full_pass):
one independent OS process per (N, realisation) task, dispatched by a Slurm array
(not Python multiprocessing -- docs/adr/0003's ARM64 ProcessPoolExecutor crash).
Each task runs ONE realisation at production length_tau via the tested
fast_model.simulate_fast and writes its own summary row -- avoids concurrent
writes across parallel tasks and avoids storing raw per-neuron traces for every
one of the (up to 50) repeats per size.

Usage (one task): python -m block2.full_pass <task_index>
"""
import csv
import json
import sys

from block2.eval import population_averaged_correlation
from block2.fast_model import simulate_fast_one
from config import load_config

RESULTS_DIR = "artifacts/block2_full_pass"


def build_full_pass_grid(sizes: list[int], repeats: list[int]) -> list[dict]:
    return [{"n": n, "realisation": r} for n, reps in zip(sizes, repeats) for r in range(reps)]


def run_one_task(
    n: int, realisation: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int,
) -> dict:
    result = simulate_fast_one(n=n, p=p, j=j, m_x=m_x, theta=theta, length_tau=length_tau,
                                sampling_rate=sampling_rate, burn_in_tau=burn_in_tau,
                                seed=seed + realisation)
    e, i, x = result[:n], result[n:2 * n], result[2 * n:]  # views, not copies
    row = {"n": n, "realisation": realisation}
    row["rate_E"], row["rate_I"], row["rate_X"] = float(e.mean()), float(i.mean()), float(x.mean())
    row["r_EE"] = population_averaged_correlation(e, e, True)
    row["r_II"] = population_averaged_correlation(i, i, True)
    row["r_EI"] = population_averaged_correlation(e, i, False)
    row["r_EX"] = population_averaged_correlation(e, x, False)
    row["r_IX"] = population_averaged_correlation(i, x, False)
    return row


def aggregate(results_dir: str = RESULTS_DIR, out_csv: str = "artifacts/block2_full_pass.csv") -> int:
    """Combines every per-task JSON row (written once all Slurm array tasks finish)
    into one CSV, sorted by (n, realisation). Run after the array job completes:
    python -m block2.full_pass --aggregate
    """
    import glob
    rows = []
    for path in sorted(glob.glob(f"{results_dir}/task_*.json")):
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: (r["n"], r["realisation"]))

    fields = ["n", "realisation", "rate_E", "rate_I", "rate_X", "r_EE", "r_II", "r_EI", "r_EX", "r_IX"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    if sys.argv[1] == "--aggregate":
        n = aggregate()
        print(f"wrote {n} rows to artifacts/block2_full_pass.csv")
        return

    task_index = int(sys.argv[1])
    config = load_config("config.yaml")
    net = config["binary_network"]
    full_pass = net["full_pass"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}

    grid = build_full_pass_grid(full_pass["sizes"], full_pass["repeats"])
    task = grid[task_index]

    row = run_one_task(
        n=task["n"], realisation=task["realisation"], p=net["connection_probability"], j=j,
        m_x=net["m_x"], theta=net["theta"], length_tau=full_pass["length_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"],
    )
    with open(f"{RESULTS_DIR}/task_{task_index:04d}.json", "w") as f:
        json.dump(row, f)
    print(f"task {task_index} (N={task['n']}, realisation={task['realisation']}) done: {row}", flush=True)


if __name__ == "__main__":
    main()
