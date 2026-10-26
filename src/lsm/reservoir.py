"""One chunk of samples through one LSM variant -> per-sample readout features.

The reservoir is the SAME network for every chunk of a variant (built from the variant's
fixed seed); only the input thinning and background noise are drawn per chunk. Features are
each E and I neuron's spike count in n_bins equal bins of every sample's window -- the
liquid state the readout sees. The "input" pseudo-variant bins the X spike train itself, the
no-reservoir baseline.
"""
import brian2 as b2
import numpy as np

from block3.model import build_network
from lsm.encode import chunk_spike_train, keep_probability, n_channels
from lsm.variants import build_connect, network_params


def thinning(config: dict, dataset: dict, train_samples: list) -> float:
    mean_events = np.mean([np.sum((s[3] >= 0) & (s[3] < dataset["window_ms"])) for s in train_samples])
    return keep_probability(mean_events, n_channels(dataset["sensor"], dataset["pool"]),
                            dataset["window_ms"], config["lsm"]["tasks"]["target_input_rate_hz"])


def bin_counts(neuron, t_ms, n_neurons, onsets, window_ms, n_bins) -> np.ndarray:
    """(n_samples, n_neurons * n_bins) spike counts, bins of window_ms / n_bins from onset."""
    bin_ms = window_ms / n_bins
    features = np.zeros((len(onsets), n_neurons, n_bins), dtype=np.float32)
    k = np.searchsorted(onsets, t_ms, side="right") - 1
    rel = t_ms - onsets[np.clip(k, 0, None)]
    ok = (k >= 0) & (rel < window_ms)
    np.add.at(features, (k[ok], neuron[ok], (rel[ok] // bin_ms).astype(np.int64)), 1.0)
    return features.reshape(len(onsets), -1)


def chunk_features(config: dict, dataset: dict, variant: str, samples: list, q: float,
                   chunk_index: int) -> tuple[np.ndarray, np.ndarray]:
    lsm, net_config = config["lsm"], config["spiking_network"]
    tasks = lsm["tasks"]
    dataset = {**dataset, "dt_ms": net_config["simulation"]["dt_ms"]}
    input_rng = np.random.default_rng(config["seed"] + tasks["seed_offset"] + chunk_index)
    idx, t_ms, onsets, labels = chunk_spike_train(samples, dataset, q, tasks["background_hz"],
                                                  net_config["burn_in_ms"], tasks["gap_ms"], input_rng)
    n_x = n_channels(dataset["sensor"], dataset["pool"])
    if variant == "input":
        return bin_counts(idx, t_ms, n_x, onsets, dataset["window_ms"], dataset["n_bins"]), labels

    params = network_params(config, variant, lsm["size"], n_x, gain_key="task_input_gain")
    params["external_input"] = {"kind": "spike_generator", "indices": idx, "times_ms": t_ms}
    net_rng = np.random.default_rng(config["seed"])  # one fixed reservoir per variant
    net = build_network(params, net_rng, connect=build_connect(config, variant, params, net_rng))
    end_ms = onsets[-1] + dataset["window_ms"] + tasks["gap_ms"]
    net.network.run(end_ms * b2.ms)
    v = np.asarray(net.group_e.V / b2.mV)
    assert not np.any(np.isnan(v)), f"NaN in {variant} reservoir, chunk {chunk_index}"

    n_e = params["populations"]["n_excitatory"]
    neuron = np.concatenate([np.asarray(net.spikes_e.i), n_e + np.asarray(net.spikes_i.i)])
    t = np.concatenate([np.asarray(net.spikes_e.t / b2.ms), np.asarray(net.spikes_i.t / b2.ms)])
    return bin_counts(neuron, t, lsm["size"], onsets, dataset["window_ms"], dataset["n_bins"]), labels
