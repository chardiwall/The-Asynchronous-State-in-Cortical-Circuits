"""Fig. 3A and 3B: what one network run records.

Both panels come from one simulation, being the same figure's two halves -- 3A the raster
over the z-scored tracking curves, 3B the pairwise-correlation histogram against its null. run_one_task reduces the same network to
the two population rates and r-bar, which is what the multi-network sweep collects.

Network: block3.model. Equations: block3.measure. Driven by block3.run.
"""
import csv
import os

import brian2 as b2
import numpy as np

from analysis import jitter_spike_times
from block3.measure import (
    pairwise_correlations,
    population_averaged_pairwise_correlation,
    population_rate_hz,
    sample_neuron_subset,
    spike_times_by_neuron,
)
from block3.model import build_network

PANEL_DIR = "artifacts/block3_panels"


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


def rate_sorted_sample(
    spike_times_by_neuron: list[list[float]], n_sample: int, rng: np.random.Generator
) -> np.ndarray:
    """A random neuron sample ordered by rate, low to high -- Fig. 3A's "sorted by rate".
    Over a fixed window the spike count orders the sample identically to the rate."""
    n_total = len(spike_times_by_neuron)
    chosen = sample_neuron_subset(n_total, min(n_sample, n_total), rng)
    return chosen[np.argsort([len(spike_times_by_neuron[i]) for i in chosen], kind="stable")]


def binned_population_activity(
    spike_times_ms, duration_ms: float, bin_ms: float
) -> np.ndarray:
    """Population spike count per bin (Fig. 3A bins at 3 ms), from a FLAT array. Summed, not
    averaged: the panel z-scores, so any constant factor is irrelevant.

    Flat because the external population never needs splitting by neuron -- ~50M spikes at
    the panel length, several GB of Python object overhead for a histogram that discards
    identity anyway.
    """
    edges = np.arange(int(round(duration_ms / bin_ms)) + 1) * bin_ms
    counts, _ = np.histogram(np.asarray(spike_times_ms, dtype=np.float64), bins=edges)
    return counts


def _z_score(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-12)


def _flat_spike_times(spike_monitor, burn_in_ms: float) -> np.ndarray:
    """Every spike after burn-in, re-zeroed, unsplit by neuron -- all the tracking needs."""
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    return t_ms[t_ms >= burn_in_ms] - burn_in_ms


def population_spikes(net, config: dict, burn_in_ms: float) -> dict:
    """Per-neuron times for E and I, which the raster and histogram need; flat for X."""
    sizes = config["spiking_network"]["populations"]
    return {
        "E": spike_times_by_neuron(net.spikes_e, sizes["n_excitatory"], burn_in_ms),
        "I": spike_times_by_neuron(net.spikes_i, sizes["n_inhibitory"], burn_in_ms),
        "X": _flat_spike_times(net.spikes_x, burn_in_ms),
    }


def write_fig3a(spikes: dict, recorded_ms: float, config: dict, rng) -> None:
    """Raster rows, one per sampled neuron, and the three z-scored tracking curves.

    The caption says "500 E (green) and I (red) neurons" without saying whether that means
    500 total or 500 each. Read here as 500 TOTAL, split in proportion to population size.
    """
    fig3a = config["spiking_network"]["fig3a"]
    n_e, n_i = len(spikes["E"]), len(spikes["I"])
    share_e = round(fig3a["raster_neurons"] * n_e / (n_e + n_i))

    with open(f"{PANEL_DIR}/fig3a_raster.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "population", "t_ms"])
        row = 0
        for population, n_sample in (("E", share_e), ("I", fig3a["raster_neurons"] - share_e)):
            for index in rate_sorted_sample(spikes[population], n_sample, rng):
                for t in spikes[population][index]:
                    writer.writerow([row, population, t])
                row += 1

    bin_ms = fig3a["tracking_bin_ms"]
    flat = {"E": np.concatenate([np.asarray(t) for t in spikes["E"]]),
            "I": np.concatenate([np.asarray(t) for t in spikes["I"]]),
            "X": spikes["X"]}
    activity = {p: _z_score(binned_population_activity(flat[p], recorded_ms, bin_ms).astype(float))
                for p in ("E", "I", "X")}
    with open(f"{PANEL_DIR}/fig3a_tracking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms", "E", "I", "X"])
        writer.writerows([[k * bin_ms, activity["E"][k], activity["I"][k], activity["X"][k]]
                          for k in range(len(activity["E"]))])


def write_fig3b(spikes: dict, recorded_ms: float, config: dict, rng) -> None:
    """Measured and jittered-surrogate pairwise r over the same subsample and count window,
    so the two histograms are directly comparable."""
    net_config = config["spiking_network"]
    analysis = config["analysis"]
    jitter_ms = net_config["fig3b"]["jitter_ms"]

    subset = sample_neuron_subset(len(spikes["E"]),
                                  min(net_config["r_bar_sample_size"], len(spikes["E"])), rng)
    trains = [spikes["E"][i] for i in subset]
    kwargs = dict(duration_ms=recorded_ms, bin_dt_ms=analysis["bin_dt_ms"],
                  window_T_ms=analysis["count_window_T_ms"])

    measured = pairwise_correlations(trains, **kwargs)
    surrogate = np.concatenate([
        pairwise_correlations([list(jitter_spike_times(t, jitter_ms, rng)) for t in trains], **kwargs)
        for _ in range(net_config["fig3b"]["n_surrogate_sets"])
    ])

    with open(f"{PANEL_DIR}/fig3b_correlations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "r"])
        writer.writerows([["measured", r] for r in measured])
        writer.writerows([["jittered", r] for r in surrogate])


def check_rate_matrix_fits(config: dict, length_s: float) -> float:
    """Raises BEFORE the network is built if Fig. 3B's rate matrix would not fit.

    pairwise_correlations stacks an (n_sample, n_windows) float64 matrix and numpy.corrcoef
    allocates a centred copy on top: 1.6GB at 200 s, 40GB at the 5000 s Fig. S6 states.
    Checking after the run would mean discovering that once a multi-day simulation had
    finished and written nothing.
    """
    net_config = config["spiking_network"]
    analysis = config["analysis"]
    recorded_ms = length_s * 1000.0 - net_config["burn_in_ms"]
    n_windows = recorded_ms / analysis["bin_dt_ms"] - analysis["count_window_T_ms"] / analysis["bin_dt_ms"]
    n_sample = min(net_config["r_bar_sample_size"], net_config["populations"]["n_excitatory"])
    gigabytes = n_sample * n_windows * 8 / 1e9
    budget = net_config["panels"]["max_rate_matrix_gb"]
    if gigabytes > budget:
        raise ValueError(
            f"Fig. 3B's rate matrix would be {gigabytes:.1f}GB at length_s={length_s} "
            f"({n_sample} neurons x {n_windows:,.0f} windows x 8 bytes), over the "
            f"{budget}GB budget in spiking_network.panels.max_rate_matrix_gb -- and "
            f"numpy.corrcoef adds a centred copy on top. Shorten the run, cut "
            f"r_bar_sample_size, coarsen analysis.bin_dt_ms, or raise the budget to match "
            f"the job's --mem. See GitHub issue #8."
        )
    return gigabytes
