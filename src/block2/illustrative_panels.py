"""Raw data for Fig. 2's illustrative single-realisation panels (2B, 2D, 2E, 2G).
block2.full_pass / full_pass_current cover 2C's ensemble statistics; these four panels
show qualitative shape instead, so they run short windows
(config.yaml binary_network.illustrative_panels) and need no DGX.

Usage: python -m block2.illustrative_panels {b,d,e,g}
"""
import csv
import os
import sys

import numpy as np

from block2.connectivity import couplings
from block2.eval import COMPONENT_KEYS, current_component_ccg
from block2.fast_model import simulate_fast_current
from block2.panel_traces import (
    population_mean_trace,
    single_cell_components_trace,
    subsample_state_trace,
)
from config import load_config

OUT_DIR = "artifacts/block2_illustrative"


def _settings(config: dict):
    """(binary_network config, j couplings, illustrative-panel windows, largest fixed N).

    2B/2E/2G are all specified at the largest of config's sizes_fixed (the paper's
    N=8192); 2D uses every entry of sizes_fixed.
    """
    net = config["binary_network"]
    return net, couplings(net), net["illustrative_panels"], net["sizes_fixed"][-1]


def _write(name: str, header: list[str], rows) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(f"{OUT_DIR}/panel_{name}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_panel_b(config: dict) -> None:
    """One representative E cell's E/I/X current components and their Total (S-Eq 7)."""
    net, j, windows, n = _settings(config)
    sr = net["sampling_rate_lagged"]
    trace = single_cell_components_trace(
        n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        window_tau=windows["panel_b_window_tau"], sampling_rate=sr,
        burn_in_tau=net["burn_in_tau"], seed=config["seed"], cell_index=0,
    )
    dt_ms = net["tau_ms"] / sr
    _write("b", ["t_ms", "E", "I", "X", "Total"],
           ([k * dt_ms, *trace[:, k]] for k in range(trace.shape[1])))


def generate_panel_d(config: dict) -> None:
    """Instantaneous population-averaged activities m_E/m_I/m_X at every fixed N."""
    net, j, windows, _ = _settings(config)
    sr = net["sampling_rate_lagged"]
    dt_ms = net["tau_ms"] / sr
    rows = []
    for n in net["sizes_fixed"]:
        trace = population_mean_trace(
            n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
            window_tau=windows["panel_d_window_tau"], sampling_rate=sr,
            burn_in_tau=net["burn_in_tau"], seed=config["seed"],
        )
        rows += [[n, k * dt_ms, *trace[:, k]] for k in range(trace.shape[1])]
    _write("d", ["n", "t_ms", "E", "I", "X"], rows)


def generate_panel_e(config: dict) -> None:
    """Fig. 2E's population-averaged CCGs of the current COMPONENTS, at every fixed N. The
    paper plots the CCGs at N=8192 and uses the insets to show the EI-Lag shrinking with N,
    which needs the smaller size too.
    """
    net, j, windows, _ = _settings(config)
    sr = net["sampling_rate_lagged"]
    dt_ms = net["tau_ms"] / sr
    max_lag = round(net["ccg"]["max_lag_ms"] / dt_ms)
    # eval owns this key vocabulary -- all nine cross-terms plus the total. Re-deriving the
    # list here would let the CSV's columns drift out of step with what the decomposition
    # actually returns. Nine, not six: c_IE(lag) == c_EI(-lag), so away from zero lag the
    # total needs c_EI + c_IE rather than 2*c_EI, which is also why Fig. 2E's inset plots
    # IE and EI separately -- that asymmetry IS the EI-Lag.
    fields = list(COMPONENT_KEYS)
    rows = []
    for n in net["sizes_fixed"]:
        components = simulate_fast_current(
            n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
            length_tau=windows["panel_e_window_tau"], sampling_rate=sr,
            burn_in_tau=net["burn_in_tau"], seed=config["seed"],
            subsample_size=net["fig2g_subsample_neurons"],
        )
        ccg = current_component_ccg(components, max_lag)
        rows += [[n, lag * dt_ms, *(ccg[f][k] for f in fields)]
                 for k, lag in enumerate(range(-max_lag, max_lag + 1))]
    _write("e", ["n", "lag_ms", *fields], rows)


def generate_panel_g(config: dict) -> None:
    """Every pairwise firing correlation r among a random E-neuron subsample."""
    net, j, windows, n = _settings(config)
    trace = subsample_state_trace(
        n=n, p=net["connection_probability"], j=j, m_x=net["m_x"], theta=net["theta"],
        window_tau=windows["panel_g_window_tau"],
        sampling_rate=net["sampling_rate_instantaneous"], burn_in_tau=net["burn_in_tau"],
        seed=config["seed"], subsample_size=net["fig2g_subsample_neurons"],
    )
    corr = np.corrcoef(trace)
    _write("g", ["r"], ([r] for r in corr[np.triu_indices_from(corr, k=1)]))


GENERATORS = {"b": generate_panel_b, "d": generate_panel_d, "e": generate_panel_e, "g": generate_panel_g}


def main():
    panel = sys.argv[1]
    GENERATORS[panel](load_config("config.yaml"))
    print(f"panel {panel} data written to {OUT_DIR}/panel_{panel}.csv")


if __name__ == "__main__":
    main()
