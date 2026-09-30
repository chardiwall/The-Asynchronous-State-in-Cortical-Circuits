"""Block 2's exploratory validation pass (session 2026-09-11, grilling Q1/Q11): fast,
small-N runs checked against docs/paper/02-binary-network.md's closed-form theory
(S-Eq 18, 28-29) -- not reproduction numbers. Also benchmarks throughput (ticks/sec),
which the deferred full-pass sizing decision (ADR 0003, Q12/Q13) needs.

Usage: python -m block2.exploratory_pass
"""
import csv
import datetime
import time

import numpy as np

from block2.connectivity import couplings
from block2.eval import population_averaged_correlation
from block2.simulate import simulate
from block2.theory import predicted_rates
from config import load_config

CSV_PATH = "artifacts/block2_exploratory_pass.csv"
CSV_FIELDS = [
    "n", "observed_E", "observed_I", "observed_X", "predicted_E", "predicted_I", "predicted_X",
    "r_EE", "r_II", "r_EI", "r_EX", "r_IX", "n_ticks", "elapsed_s", "ticks_per_sec", "timestamp",
]


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


def write_csv(rows: list[dict], path: str = CSV_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_config("config.yaml")
    rows = run_exploratory_pass(config)
    write_csv(rows)
    print(f"wrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
