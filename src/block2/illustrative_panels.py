"""Generates the raw data for Fig. 2B/2D/2E/2G's illustrative single-realisation
panels (job 1040 / full_pass_current only cover 2C's ensemble statistics).
Short, cheap runs (config.yaml binary_network.illustrative_panels) -- run
locally, no DGX/Slurm needed.

Usage: python -m block2.illustrative_panels {b,d,e,g}
"""
import csv
import sys

import numpy as np

from block2.eval import population_averaged_ccg
from block2.fast_model import simulate_fast_current
from block2.panel_traces import (
    population_mean_trace,
    single_cell_components_trace,
    subsample_state_trace,
)
from config import load_config

OUT_DIR = "artifacts/block2_illustrative"


def generate_panel_b(config: dict) -> None:
    net = config["binary_network"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}
    sr = net["sampling_rate_lagged"]
    trace = single_cell_components_trace(
        n=8192, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        window_tau=net["illustrative_panels"]["panel_b_window_tau"], sampling_rate=sr,
        burn_in_tau=net["burn_in_tau"], seed=config["seed"], cell_index=0,
    )
    dt_ms = net["tau_ms"] / sr
    with open(f"{OUT_DIR}/panel_b.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms", "E", "I", "X", "Total"])
        for k in range(trace.shape[1]):
            writer.writerow([k * dt_ms, *trace[:, k]])


def generate_panel_d(config: dict) -> None:
    net = config["binary_network"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}
    sr = net["sampling_rate_lagged"]
    dt_ms = net["tau_ms"] / sr
    with open(f"{OUT_DIR}/panel_d.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "t_ms", "E", "I", "X"])
        for n in net["sizes_fixed"]:
            trace = population_mean_trace(
                n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
                window_tau=net["illustrative_panels"]["panel_d_window_tau"], sampling_rate=sr,
                burn_in_tau=net["burn_in_tau"], seed=config["seed"],
            )
            for k in range(trace.shape[1]):
                writer.writerow([n, k * dt_ms, *trace[:, k]])


def generate_panel_e(config: dict) -> None:
    net = config["binary_network"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}
    sr = net["sampling_rate_lagged"]
    dt_ms = net["tau_ms"] / sr
    max_lag = round(net["ccg"]["max_lag_ms"] / dt_ms)
    current = simulate_fast_current(
        n=8192, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        length_tau=net["illustrative_panels"]["panel_e_window_tau"], sampling_rate=sr,
        burn_in_tau=net["burn_in_tau"], seed=config["seed"],
        subsample_size=net["fig2g_subsample_neurons"],
    )
    ccg_ee = population_averaged_ccg(current["E"], current["E"], max_lag, True)
    ccg_ii = population_averaged_ccg(current["I"], current["I"], max_lag, True)
    ccg_ei = population_averaged_ccg(current["E"], current["I"], max_lag, False)
    ccg_total = ccg_ee + ccg_ii + 2 * ccg_ei
    with open(f"{OUT_DIR}/panel_e.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lag_ms", "ccg_EE", "ccg_II", "ccg_EI", "ccg_total"])
        for k, lag in enumerate(range(-max_lag, max_lag + 1)):
            writer.writerow([lag * dt_ms, ccg_ee[k], ccg_ii[k], ccg_ei[k], ccg_total[k]])


def generate_panel_g(config: dict) -> None:
    net = config["binary_network"]
    j = {pair: net["couplings"][f"j_{pair}"] for pair in ("EE", "EI", "EX", "IE", "II", "IX")}
    trace = subsample_state_trace(
        n=8192, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        window_tau=net["illustrative_panels"]["panel_g_window_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"], subsample_size=net["fig2g_subsample_neurons"],
    )
    corr = np.corrcoef(trace)
    pairwise_r = corr[np.triu_indices_from(corr, k=1)]
    with open(f"{OUT_DIR}/panel_g.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["r"])
        writer.writerows([[r] for r in pairwise_r])


GENERATORS = {"b": generate_panel_b, "d": generate_panel_d, "e": generate_panel_e, "g": generate_panel_g}


def main():
    config = load_config("config.yaml")
    GENERATORS[sys.argv[1]](config)
    print(f"panel {sys.argv[1]} data written to {OUT_DIR}/panel_{sys.argv[1]}.csv")


if __name__ == "__main__":
    main()
