"""Block 3's exploratory pass (session 2026-09-14, PROGRESS.md Phase 1): local smoke test at
full paper-scale N, short duration, single network -- proves the model and the existing
shared analysis pipeline (src/analysis.py) are wired correctly (NaN-free, non-zero,
non-runaway rates; a roughly-shaped r_bar). Not a statistically solid r_bar estimate -- that
is Fig. 3A-B's full pass (Phase 2, a later session).

Usage: python -m block3.exploratory_pass
"""
import datetime
import json

import brian2 as b2
import numpy as np

from analysis import spike_count_correlation
from block3.model import build_network
from config import load_config

LOG_PATH = "artifacts/block3_exploratory_pass.log"
N_PAIRS_FOR_R_SAMPLE = 25  # exploratory sample only, not Fig. 3B's real pair count


def _population_rate_hz(spike_monitor, n_neurons, burn_in_ms, duration_ms) -> float:
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    n_spikes_after_burn_in = int(np.sum(t_ms >= burn_in_ms))
    recorded_s = (duration_ms - burn_in_ms) / 1000.0
    return n_spikes_after_burn_in / (n_neurons * recorded_s)


def _spike_times_by_neuron(spike_monitor, n_neurons, burn_in_ms) -> list[list[float]]:
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    neuron_index = np.asarray(spike_monitor.i)
    after_burn_in = t_ms >= burn_in_ms
    return [
        list(t_ms[after_burn_in & (neuron_index == n)] - burn_in_ms) for n in range(n_neurons)
    ]


def run_exploratory_pass(config: dict) -> dict:
    net_config = config["spiking_network"]
    exploratory = net_config["exploratory"]
    length_ms = exploratory["length_s"] * 1000.0
    burn_in_ms = net_config["burn_in_ms"]

    params = {**net_config, "dt_ms": net_config["simulation"]["dt_ms"]}
    rng = np.random.default_rng(config["seed"])

    net = build_network(params, rng)
    net.network.run(length_ms * b2.ms)

    v_e = np.asarray(net.group_e.V / b2.mV)
    v_i = np.asarray(net.group_i.V / b2.mV)
    assert not np.any(np.isnan(v_e)), "NaN in E population membrane potential"
    assert not np.any(np.isnan(v_i)), "NaN in I population membrane potential"

    n_e = net_config["populations"]["n_excitatory"]
    n_i = net_config["populations"]["n_inhibitory"]
    rate_e_hz = _population_rate_hz(net.spikes_e, n_e, burn_in_ms, length_ms)
    rate_i_hz = _population_rate_hz(net.spikes_i, n_i, burn_in_ms, length_ms)

    analysis_config = config["analysis"]
    recorded_ms = length_ms - burn_in_ms
    n_sample_neurons = min(n_e, 2 * N_PAIRS_FOR_R_SAMPLE)
    ee_spike_times = _spike_times_by_neuron(net.spikes_e, n_sample_neurons, burn_in_ms)
    r_values = [
        spike_count_correlation(
            ee_spike_times[2 * k], ee_spike_times[2 * k + 1], recorded_ms,
            analysis_config["bin_dt_ms"], analysis_config["count_window_T_ms"],
        )
        for k in range(n_sample_neurons // 2)
        if len(ee_spike_times[2 * k]) > 1 and len(ee_spike_times[2 * k + 1]) > 1
    ]
    r_values = [r for r in r_values if np.isfinite(r)]

    return {
        "n_excitatory": n_e, "n_inhibitory": n_i,
        "length_s": exploratory["length_s"], "burn_in_ms": burn_in_ms,
        "rate_excitatory_hz": rate_e_hz, "rate_inhibitory_hz": rate_i_hz,
        "r_bar_EE_sample": float(np.mean(r_values)) if r_values else None,
        "n_pairs_used": len(r_values),
        "nan_free": True,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def main():
    config = load_config("config.yaml")
    result = run_exploratory_pass(config)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
