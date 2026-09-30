"""Block 3's full pass (Fig. 3A-B, PROGRESS.md Phase 2): one independent Slurm array task
per network realisation (docs/adr/0003's pattern -- no in-process pool, one OS process per
task via a Slurm array, run_one_task.slurm). Each task builds and runs one full paper-scale
network, computes its summary statistics, and returns/writes one row -- mirrors
block2/full_pass.py's run_one_task shape.

A short run doubles as the smoke test the separate exploratory_pass.py used to be:
`--seconds 5` runs the same code path at the same paper-scale N over a shorter window,
so it is a true subset of the full pass rather than a second implementation of it.

Usage (one task): python -m block3.full_pass <network_index> [--seconds N]
Usage (aggregate): python -m block3.full_pass --aggregate
"""
import csv
import sys

import brian2 as b2
import numpy as np

from block3.eval import (
    population_averaged_pairwise_correlation,
    population_rate_hz,
    sample_neuron_subset,
    spike_times_by_neuron,
)
from block3.model import build_network
from config import load_config

RESULTS_DIR = "artifacts/block3_full_pass"
CSV_FIELDS = ["network_index", "rate_excitatory_hz", "rate_inhibitory_hz", "r_bar_EE", "nan_free"]


def parse_length_s(argv: list[str], default: float) -> float:
    """The `--seconds N` override, parsed strictly.

    A malformed invocation RAISES rather than falling back to `default`. That matters more
    than it looks: the flag exists so a 5-second smoke test can be run before a Slurm array
    is submitted whose tasks each cost many hours, and a silent fallback would turn the
    check into the very paper-scale run it was meant to de-risk -- while appearing to
    succeed.
    """
    if len(argv) <= 2:
        return default
    if argv[2] != "--seconds":
        raise ValueError(f"unrecognised argument {argv[2]!r}; the only option is --seconds N")
    if len(argv) < 4:
        raise ValueError("--seconds needs a value, e.g. --seconds 5")
    return float(argv[3])


def run_one_task(network_index: int, config: dict, duration_ms: float) -> dict:
    net_config = config["spiking_network"]
    burn_in_ms = net_config["burn_in_ms"]
    analysis_config = config["analysis"]

    params = {**net_config, "dt_ms": net_config["simulation"]["dt_ms"]}
    rng = np.random.default_rng(config["seed"] + network_index)

    net = build_network(params, rng)
    net.network.run(duration_ms * b2.ms)

    v_e = np.asarray(net.group_e.V / b2.mV)
    v_i = np.asarray(net.group_i.V / b2.mV)
    nan_free = not (np.any(np.isnan(v_e)) or np.any(np.isnan(v_i)))
    assert nan_free, f"NaN in network {network_index}'s membrane potential"

    n_e = net_config["populations"]["n_excitatory"]
    n_i = net_config["populations"]["n_inhibitory"]
    rate_e_hz = population_rate_hz(net.spikes_e, n_e, burn_in_ms, duration_ms)
    rate_i_hz = population_rate_hz(net.spikes_i, n_i, burn_in_ms, duration_ms)

    sample_size = net_config["r_bar_sample_size"]
    subset = sample_neuron_subset(n_e, min(sample_size, n_e), rng)
    all_ee_spikes = spike_times_by_neuron(net.spikes_e, n_e, burn_in_ms)
    subset_spikes = [all_ee_spikes[i] for i in subset]
    r_bar_ee = population_averaged_pairwise_correlation(
        subset_spikes, duration_ms - burn_in_ms, analysis_config["bin_dt_ms"],
        analysis_config["count_window_T_ms"],
    )

    return {
        "network_index": network_index,
        "rate_excitatory_hz": rate_e_hz, "rate_inhibitory_hz": rate_i_hz,
        "r_bar_EE": r_bar_ee, "nan_free": nan_free,
    }


def aggregate(results_dir: str = RESULTS_DIR, out_csv: str = "artifacts/block3_full_pass.csv") -> int:
    """Combines every per-task CSV row (written once all Slurm array tasks finish) into one
    CSV, sorted by network_index. Run after the array job completes:
    python -m block3.full_pass --aggregate
    """
    import glob

    rows = []
    for path in sorted(glob.glob(f"{results_dir}/network_*.csv")):
        with open(path, newline="") as f:
            rows.append(next(csv.DictReader(f)))
    rows.sort(key=lambda r: int(r["network_index"]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    if sys.argv[1] == "--aggregate":
        n = aggregate()
        print(f"wrote {n} rows to artifacts/block3_full_pass.csv")
        return

    network_index = int(sys.argv[1])
    config = load_config("config.yaml")
    duration_ms = parse_length_s(
        sys.argv, config["spiking_network"]["simulation"]["length_s"]) * 1000.0

    row = run_one_task(network_index, config, duration_ms)

    path = f"{RESULTS_DIR}/network_{network_index}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {path}: {row}")


if __name__ == "__main__":
    main()
