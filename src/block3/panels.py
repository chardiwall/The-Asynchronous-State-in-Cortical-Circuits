"""Fig. 3A and 3B's data, from one network run.

3A is the raster of a rate-sorted neuron sample plus the z-scored population activities
that show E, I and X tracking each other; 3B is the histogram of pairwise spike-count
correlations together with its jittered-surrogate null. Both come out of the same
simulation because they are the same figure's two halves.

Usage: python -m block3.panels [--seconds N]
"""
import csv
import os
import sys

import brian2 as b2
import numpy as np

from analysis import jitter_spike_times
from block3.eval import pairwise_correlations, sample_neuron_subset, spike_times_by_neuron
from block3.full_pass import parse_length_s
from block3.model import build_network
from config import load_config

OUT_DIR = "artifacts/block3_panels"


def rate_sorted_sample(
    spike_times_by_neuron: list[list[float]], n_sample: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of a random neuron sample ordered by firing rate, low to high -- Fig. 3A's
    "500 E and I neurons sorted by rate". Over a fixed window the spike COUNT is a
    monotone function of the rate, so counts order the sample identically.
    """
    n_total = len(spike_times_by_neuron)
    chosen = sample_neuron_subset(n_total, min(n_sample, n_total), rng)
    return chosen[np.argsort([len(spike_times_by_neuron[i]) for i in chosen], kind="stable")]


def binned_population_activity(
    spike_times_ms, duration_ms: float, bin_ms: float
) -> np.ndarray:
    """Total spike count of the population in each bin (Fig. 3A's bin size is 3 ms), from a
    FLAT array of every spike the population emitted. Summed, not averaged: the z-scoring
    the panel applies makes any constant factor irrelevant, so this stays a raw count.

    Flat rather than per-neuron because the external population never needs splitting by
    neuron -- at the panel run's length that is ~50M spikes, and materialising them as
    Python lists costs several GB of object overhead for a histogram that discards the
    identity anyway.
    """
    edges = np.arange(int(round(duration_ms / bin_ms)) + 1) * bin_ms
    counts, _ = np.histogram(np.asarray(spike_times_ms, dtype=np.float64), bins=edges)
    return counts


def _z_score(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-12)


def _flat_spike_times(spike_monitor, burn_in_ms: float) -> np.ndarray:
    """Every spike the population emitted after burn-in, re-zeroed, without splitting by
    neuron -- all the tracking curve needs.
    """
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    return t_ms[t_ms >= burn_in_ms] - burn_in_ms


def _population_spikes(net, config: dict, burn_in_ms: float) -> dict:
    """Per-neuron spike times for E and I, which the raster and the histogram both need,
    and a flat array for X, which only ever appears binned.
    """
    sizes = config["spiking_network"]["populations"]
    return {
        "E": spike_times_by_neuron(net.spikes_e, sizes["n_excitatory"], burn_in_ms),
        "I": spike_times_by_neuron(net.spikes_i, sizes["n_inhibitory"], burn_in_ms),
        "X": _flat_spike_times(net.spikes_x, burn_in_ms),
    }


def write_fig3a(spikes: dict, recorded_ms: float, config: dict, rng) -> None:
    """Raster rows (one per sampled neuron) and the three z-scored tracking curves.

    The caption says "500 E (green) and I (red) neurons" without saying whether that is
    500 total or 500 each; read here as 500 TOTAL split between E and I in proportion to
    their population sizes (config.yaml records this reading).
    """
    fig3a = config["spiking_network"]["fig3a"]
    n_e, n_i = len(spikes["E"]), len(spikes["I"])
    share_e = round(fig3a["raster_neurons"] * n_e / (n_e + n_i))

    with open(f"{OUT_DIR}/fig3a_raster.csv", "w", newline="") as f:
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
    with open(f"{OUT_DIR}/fig3a_tracking.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ms", "E", "I", "X"])
        writer.writerows([[k * bin_ms, activity["E"][k], activity["I"][k], activity["X"][k]]
                          for k in range(len(activity["E"]))])


def write_fig3b(spikes: dict, recorded_ms: float, config: dict, rng) -> None:
    """Measured and jittered-surrogate pairwise r, over the same E subsample and the same
    T=50 ms count window, so the two histograms are directly comparable.
    """
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

    with open(f"{OUT_DIR}/fig3b_correlations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "r"])
        writer.writerows([["measured", r] for r in measured])
        writer.writerows([["jittered", r] for r in surrogate])


def check_rate_matrix_fits(config: dict, length_s: float) -> float:
    """Raises BEFORE the network is built if Fig. 3B's rate matrix would not fit.

    eval.pairwise_correlations stacks an (n_sample, n_windows) float64 matrix and hands it
    to numpy.corrcoef, which allocates a centred copy on top. At 200 s that is 1.6GB; at the
    5000 s Fig. S6 states it is 40GB. Checking after the run would mean discovering it once
    the multi-day simulation had finished and written nothing.
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


def main():
    config = load_config("config.yaml")
    net_config = config["spiking_network"]
    # Fig. 3B's own length, not the Fig. 3A-B statistics run's: the surrogate histogram's
    # width is pure estimator noise, so 200 s makes the null as wide as the measurement.
    # argv is padded so --seconds sits where parse_length_s expects it: this entry point
    # takes no task index, unlike full_pass.
    length_s = parse_length_s(["panels", "-", *sys.argv[1:]], net_config["panels"]["length_s"])
    duration_ms = length_s * 1000.0
    burn_in_ms = net_config["burn_in_ms"]

    projected_gb = check_rate_matrix_fits(config, length_s)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fig. 3B rate matrix will be ~{projected_gb:.2f}GB", flush=True)

    rng = np.random.default_rng(config["seed"])
    net = build_network({**net_config, "dt_ms": net_config["simulation"]["dt_ms"]}, rng)
    net.network.run(duration_ms * b2.ms)

    spikes = _population_spikes(net, config, burn_in_ms)
    recorded_ms = duration_ms - burn_in_ms
    write_fig3a(spikes, recorded_ms, config, rng)
    write_fig3b(spikes, recorded_ms, config, rng)
    print(f"Fig. 3A/3B data written to {OUT_DIR}/ ({length_s}s run)")


if __name__ == "__main__":
    main()
