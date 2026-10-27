"""Correlations of a liquid while it is DRIVEN by task input, split into a stimulus-locked
part and a trial-to-trial part. The asynchrony claim concerns the second.

The first lsm.reliability.n_samples TEST samples run twice through the same reservoir with
independent input noise (thinning and background; seeds lsm.reliability.noise_streams). With
x_k = s + n_k for repeat k (signal s, noise n_k independent of each other and across k):

  cov(x1_i, x2_j)               = S_ij      the signal covariance
  cov(x1_i - x2_i, x1_j - x2_j) = 2 N_ij    the noise covariance

  r_total   mean pairwise Pearson r of E spike counts in T-bins, within a repeat
  r_signal  mean S_ij / sqrt(S_ii S_jj), over cells with S_ii > 0
  r_noise   mean pairwise r of the repeat differences, i.e. of N
  c_*_T     block 2's current decomposition (M-Eq 3) of the T-averaged currents (repeat 1)
  c_*_noise the same decomposition of the repeat differences of those currents

    python -m lsm.driven <dataset> <variant>    -> <features>/<dataset>/driven_<variant>.json
"""
import json
import sys

import brian2 as b2
import numpy as np

from block2.measure import current_component_correlations


def _mean_offdiagonal(c: np.ndarray) -> float:
    n = len(c)
    return float((c.sum() - np.trace(c)) / (n * (n - 1))) if n > 1 else float("nan")


def repeat_correlations(first: np.ndarray, second: np.ndarray) -> dict:
    """first, second: (cells, bins) counts of the same cells in two repeats."""
    active = (first.std(axis=1) > 0) & (second.std(axis=1) > 0)
    a, b = first[active], second[active]
    total = 0.5 * (_mean_offdiagonal(np.corrcoef(a)) + _mean_offdiagonal(np.corrcoef(b)))
    diff = a - b
    noise = _mean_offdiagonal(np.corrcoef(diff[diff.std(axis=1) > 0]))
    a0, b0 = a - a.mean(axis=1, keepdims=True), b - b.mean(axis=1, keepdims=True)
    cross = a0 @ b0.T / a.shape[1]
    cross = 0.5 * (cross + cross.T)
    s = np.diag(cross)
    keep = s > 0
    signal = _mean_offdiagonal(cross[np.ix_(keep, keep)] / np.sqrt(np.outer(s[keep], s[keep])))
    return {"r_total": total, "r_signal": signal, "r_noise": noise}


def driven_row(config: dict, dataset: dict, variant: str, samples: list, q: float) -> dict:
    from lsm.asynchrony import boxcar
    from lsm.reservoir import chunk_network

    lsm, burn_in_ms = config["lsm"], config["spiking_network"]["burn_in_ms"]
    t_ms, sample_ms = config["analysis"]["count_window_T_ms"], lsm["asynchrony"]["current_sample_ms"]
    counts, currents, rates = [], [], []
    for seed in lsm["reliability"]["noise_streams"]:
        net, onsets, labels, end_ms = chunk_network(config, dataset, variant, samples, q, seed)
        n_e, n_i = len(net.group_e), len(net.group_i)
        cells = np.arange(min(lsm["asynchrony"]["n_current_cells"], n_e))
        monitor = b2.StateMonitor(net.group_e, ["I_E", "I_I", "I_X"], record=cells, dt=sample_ms * b2.ms)
        net.network.add(monitor)
        net.network.run(end_ms * b2.ms)
        keep = np.asarray(monitor.t / b2.ms) >= burn_in_ms
        components = np.stack([np.asarray(getattr(monitor, k) / b2.nA)[:, keep] for k in ("I_E", "I_I", "I_X")])
        assert not np.any(np.isnan(components)), f"NaN in {variant} driven run"
        currents.append(boxcar(components, int(round(t_ms / sample_ms))))
        edges = np.arange(burn_in_ms, end_ms + 1e-9, t_ms)
        spike_t = np.asarray(net.spikes_e.t / b2.ms)
        counts.append(np.histogram2d(np.asarray(net.spikes_e.i), spike_t, bins=[np.arange(n_e + 1), edges])[0])
        seconds = (end_ms - burn_in_ms) / 1000.0
        rates.append((np.sum(spike_t >= burn_in_ms) / (n_e * seconds),
                      np.sum(np.asarray(net.spikes_i.t / b2.ms) >= burn_in_ms) / (n_i * seconds)))
    row = {"variant": variant, "n_samples": len(samples),
           "rate_excitatory_hz": float(np.mean([r[0] for r in rates])),
           "rate_inhibitory_hz": float(np.mean([r[1] for r in rates]))}
    row.update(repeat_correlations(*counts))
    row.update({f"{k}_T": v for k, v in current_component_correlations(currents[0]).items()})
    row.update({f"{k}_noise": v for k, v in current_component_correlations(currents[0] - currents[1]).items()})
    return row


def main(name: str, variant: str) -> dict:
    from config import load_config, output_path
    from lsm.reservoir import thinning
    from lsm.run import _load, task_references

    config = load_config("config.yaml")
    dataset = config["lsm"]["datasets"][name]
    references, n_train = task_references(config, name)
    n = config["lsm"]["reliability"]["n_samples"]
    # the thinning scalar from training samples only, exactly as for the features
    q = thinning(config, dataset, _load(dataset, references[:min(n_train, config["lsm"]["tasks"]["thinning_samples"])]))
    row = {"dataset": name, **driven_row(config, dataset, variant, _load(dataset, references[n_train:n_train + n]), q)}
    path = f"{output_path(config, 'lsm_features_dir')}/{name}/driven_{variant}.json"
    with open(path, "w") as f:
        json.dump(row, f, indent=1)
    print(row, flush=True)
    return row


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
