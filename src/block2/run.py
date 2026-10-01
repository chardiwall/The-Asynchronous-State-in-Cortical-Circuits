"""Block 2's one entry point. Every Fig. 2 run is a subcommand here.

    python -m block2.run explore              small-N pass checked against the closed form
    python -m block2.run sweep <task_index>    Fig. 2C's r-bar sweep, one Slurm array task
    python -m block2.run current <task_index>  Fig. 2C's current-component sweep
    python -m block2.run aggregate             both sweeps' per-task rows -> CSVs
    python -m block2.run panels {b,d,e,g}      the illustrative panels' raw data
    python -m block2.run plot                  every figure from whatever CSVs exist

The sweeps run one OS process per (N, realisation) task via a Slurm array rather than an
in-process pool (which copied real memory per worker -- OOM at N=8192 -- and whose
ProcessPoolExecutor misbehaved on the cluster's ARM64 nodes), so `sweep` and `current`
take a flat task index and write one row each; `aggregate` stitches them together
afterwards.
"""
import csv
import datetime
import sys
import time

import numpy as np

from block2.connectivity import couplings
from block2.measure import (
    current_component_correlations,
    population_averaged_correlation,
)
from block2.model import predicted_rates, simulate
from config import load_config, output_path
from lib.glauber import simulate_fast_current, simulate_fast_one
from lib.tasks import aggregate_task_rows, size_repeat_grid, write_task_row

SWEEP_FIELDS = ["n", "realisation", "rate_E", "rate_I", "rate_X",
                "r_EE", "r_II", "r_EI", "r_EX", "r_IX"]
CURRENT_FIELDS = ["n", "realisation", "c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total"]
EXPLORE_FIELDS = [
    "n", "observed_E", "observed_I", "observed_X", "predicted_E", "predicted_I", "predicted_X",
    "r_EE", "r_II", "r_EI", "r_EX", "r_IX", "n_ticks", "elapsed_s", "ticks_per_sec", "timestamp",
]


def task_parameters(task_index: int):
    """(config, binary_network config, couplings, this task's {n, realisation}). Both sweeps
    walk the same grid, so they dispatch identically.
    """
    config = load_config("config.yaml")
    net = config["binary_network"]
    grid = size_repeat_grid(net["full_pass"]["sizes"], net["full_pass"]["repeats"])
    return config, net, couplings(net), grid[task_index]


def run_one_size(n: int, config: dict) -> dict:
    net = config["binary_network"]
    exploratory = net["exploratory"]
    j = couplings(net)

    n_realisations = exploratory["n_realisations"]
    length_tau = exploratory["length_tau"]
    sampling_rate = net["sampling_rate_instantaneous"]
    burn_in_tau = net["burn_in_tau"]

    t0 = time.time()
    result = simulate(
        n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        length_tau=length_tau, sampling_rate=sampling_rate, n_realisations=n_realisations,
        burn_in_tau=burn_in_tau, seed=config["seed"],
    )
    elapsed_s = time.time() - t0

    activity = result.activity
    assert not any(np.isnan(activity[pop]).any() for pop in ("E", "I", "X")), f"NaN in N={n} activity"

    observed = {pop: float(activity[pop].mean()) for pop in ("E", "I", "X")}
    predicted = predicted_rates(p=net["connection_probability"], j=j, m_x=net["m_x"])

    correlations = {"EE": [], "II": [], "EI": [], "EX": [], "IX": []}
    for r in range(n_realisations):
        correlations["EE"].append(population_averaged_correlation(activity["E"][r], activity["E"][r], True))
        correlations["II"].append(population_averaged_correlation(activity["I"][r], activity["I"][r], True))
        correlations["EI"].append(population_averaged_correlation(activity["E"][r], activity["I"][r], False))
        correlations["EX"].append(population_averaged_correlation(activity["E"][r], activity["X"][r], False))
        correlations["IX"].append(population_averaged_correlation(activity["I"][r], activity["X"][r], False))

    ticks_per_tau = 3 * n
    ticks_per_sample = ticks_per_tau // sampling_rate
    n_ticks = n_realisations * (burn_in_tau * ticks_per_tau + length_tau * sampling_rate * ticks_per_sample)

    print(f"N={n}: observed E/I/X = {observed['E']:.4f}/{observed['I']:.4f}/{observed['X']:.4f}, "
          f"predicted E/I/X = {predicted['E']:.4f}/{predicted['I']:.4f}/{predicted['X']:.4f}, "
          f"elapsed {elapsed_s:.1f}s, {n_ticks / elapsed_s:.0f} ticks/s", flush=True)

    return {
        "n": n,
        "observed_E": observed["E"], "observed_I": observed["I"], "observed_X": observed["X"],
        "predicted_E": predicted["E"], "predicted_I": predicted["I"], "predicted_X": predicted["X"],
        "r_EE": float(np.mean(correlations["EE"])), "r_II": float(np.mean(correlations["II"])),
        "r_EI": float(np.mean(correlations["EI"])), "r_EX": float(np.mean(correlations["EX"])),
        "r_IX": float(np.mean(correlations["IX"])),
        "n_ticks": n_ticks, "elapsed_s": elapsed_s, "ticks_per_sec": n_ticks / elapsed_s,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def run_exploratory_pass(config: dict) -> list[dict]:
    return [run_one_size(n, config) for n in config["binary_network"]["exploratory"]["sizes"]]


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


def run_one_current_task(
    n: int, realisation: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int, subsample_size: int,
) -> dict:
    components = simulate_fast_current(n=n, p=p, j=j, m_x=m_x, theta=theta,
                                        length_tau=length_tau, sampling_rate=sampling_rate,
                                        burn_in_tau=burn_in_tau, seed=seed + realisation,
                                        subsample_size=subsample_size)
    return {"n": n, "realisation": realisation, **current_component_correlations(components)}


def main():
    command = sys.argv[1]
    if command == "explore":
        config = load_config("config.yaml")
        rows = [run_one_size(n, config) for n in config["binary_network"]["exploratory"]["sizes"]]
        explore_csv = output_path(config, "block2_explore_csv")
        with open(explore_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXPLORE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows to {explore_csv}")

    elif command in ("sweep", "current"):
        index = int(sys.argv[2])
        config, net, j, task = task_parameters(index)
        shared = dict(n=task["n"], realisation=task["realisation"],
                      p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
                      length_tau=net["full_pass"]["length_tau"],
                      sampling_rate=net["sampling_rate_instantaneous"],
                      burn_in_tau=net["burn_in_tau"], seed=config["seed"])
        if command == "sweep":
            write_task_row(output_path(config, "block2_sweep_dir"), f"{index:04d}",
                           run_one_task(**shared))
        else:
            write_task_row(output_path(config, "block2_current_dir"), f"{index:04d}",
                           run_one_current_task(**shared,
                                                subsample_size=net["fig2g_subsample_neurons"]))

    elif command == "aggregate":
        config = load_config("config.yaml")
        by_realisation = lambda r: (r["n"], r["realisation"])
        for directory, out, fields in (
            (output_path(config, "block2_sweep_dir"),
             output_path(config, "block2_sweep_csv"), SWEEP_FIELDS),
            (output_path(config, "block2_current_dir"),
             output_path(config, "block2_current_csv"), CURRENT_FIELDS),
        ):
            print(f"{out}: {aggregate_task_rows(directory, out, fields, by_realisation)} rows")

    elif command == "panels":
        from block2 import panels

        panel = sys.argv[2]
        panels.GENERATORS[panel](load_config("config.yaml"))
        print(f"panel {panel} data written")

    elif command == "plot":
        from block2 import plot, plot_sweep

        plot_sweep.plot_fig2c()
        plot.plot_fig2b(); plot.plot_fig2d(); plot.plot_fig2e(); plot.plot_fig2g()
        print("wrote fig2b/c/d/e/g.png")

    else:
        raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
