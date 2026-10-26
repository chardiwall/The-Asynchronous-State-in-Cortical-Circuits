"""Is the paper's asynchronous state present in an LSM variant? One background-only run
(block 3's Poisson X at 2.5 Hz, no task input) measured with the paper's own estimators:

  rates           population means after burn-in
  r_bar, sigma_r  spike-count correlation over E pairs, T = 50 ms (S-Eq 34-37), and sigma_r
                  of a +-500 ms jittered surrogate -- the estimator-noise null of Fig. 3B
  c components    the E/I/X current decomposition of M-Eq 3, with block 2's common-
                  normalisation estimator so c = c_EE + c_II + c_XX + 2(c_EI + c_EX + c_IX),
                  on 1 ms currents and (suffix _T) on currents averaged over T = 50 ms
  tracking        zero-lag correlation of z-scored population activity, E vs I and E vs X
                  (Fig. 3A's 3 ms bins): the mechanism -- inhibition tracking excitation
"""
import brian2 as b2
import numpy as np

from analysis import jitter_spike_times
from block2.measure import current_component_correlations
from block3.figures_ab import binned_population_activity
from block3.measure import (pairwise_correlations, population_rate_hz, sample_neuron_subset,
                            spike_times_by_neuron)
from block3.model import build_network
from lsm.variants import build_connect, network_params


def _tracking(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def build_background(config: dict, variant: str, n: int, network_index: int):
    lsm = config["lsm"]
    n_x = round(lsm["external_per_recurrent"] * n)
    params = network_params(config, variant, n, n_x)
    rng = np.random.default_rng(config["seed"] + 1000 * network_index + n)
    return build_network(params, rng, connect=build_connect(config, variant, params, rng)), params, rng


def run_background(config: dict, variant: str, n: int, network_index: int, length_s: float,
                   record_currents: bool = True) -> dict:
    net_config, analysis, lsm = config["spiking_network"], config["analysis"], config["lsm"]
    burn_in_ms, duration_ms = net_config["burn_in_ms"], length_s * 1000.0
    net, params, rng = build_background(config, variant, n, network_index)
    sizes = params["populations"]

    if record_currents:
        cells = sample_neuron_subset(sizes["n_excitatory"],
                                     min(lsm["asynchrony"]["n_current_cells"], sizes["n_excitatory"]), rng)
        monitor = b2.StateMonitor(net.group_e, ["I_E", "I_I", "I_X"], record=list(cells),
                                  dt=lsm["asynchrony"]["current_sample_ms"] * b2.ms)
        net.network.add(monitor)
    net.network.run(duration_ms * b2.ms)

    v = np.concatenate([np.asarray(net.group_e.V / b2.mV), np.asarray(net.group_i.V / b2.mV)])
    assert not np.any(np.isnan(v)), f"NaN in {variant} N={n} network {network_index}"
    row = {"variant": variant, "n": n, "network_index": network_index, "length_s": length_s,
           "rate_excitatory_hz": population_rate_hz(net.spikes_e, sizes["n_excitatory"], burn_in_ms, duration_ms),
           "rate_inhibitory_hz": population_rate_hz(net.spikes_i, sizes["n_inhibitory"], burn_in_ms, duration_ms)}

    recorded_ms = duration_ms - burn_in_ms
    spikes_e = spike_times_by_neuron(net.spikes_e, sizes["n_excitatory"], burn_in_ms)
    subset = sample_neuron_subset(len(spikes_e), min(net_config["r_bar_sample_size"], len(spikes_e)), rng)
    trains = [spikes_e[i] for i in subset]
    kwargs = dict(duration_ms=recorded_ms, bin_dt_ms=analysis["bin_dt_ms"],
                  window_T_ms=analysis["count_window_T_ms"])
    r = pairwise_correlations(trains, **kwargs)
    jitter_ms = net_config["fig3b"]["jitter_ms"]
    r_null = pairwise_correlations([list(jitter_spike_times(t, jitter_ms, rng)) for t in trains], **kwargs)
    row.update(r_bar=float(np.nanmean(r)), sigma_r=float(np.nanstd(r)),
               sigma_r_jittered=float(np.nanstd(r_null)), n_pairs_defined=int(np.sum(~np.isnan(r))))

    bin_ms = net_config["fig3a"]["tracking_bin_ms"]
    activity = {}
    for name, monitor_spikes in (("E", net.spikes_e), ("I", net.spikes_i), ("X", net.spikes_x)):
        t = np.asarray(monitor_spikes.t / b2.ms)
        activity[name] = binned_population_activity(t[t >= burn_in_ms] - burn_in_ms, recorded_ms, bin_ms).astype(float)
    row.update(tracking_EI=_tracking(activity["E"], activity["I"]),
               tracking_EX=_tracking(activity["E"], activity["X"]))

    if record_currents:
        keep = np.asarray(monitor.t / b2.ms) >= burn_in_ms
        components = np.stack([np.asarray(getattr(monitor, k) / b2.nA)[:, keep]
                               for k in ("I_E", "I_I", "I_X")])
        row.update(current_component_correlations(components))
        # Spikes respond to the current low-passed over ~T, not to its 1 ms shot noise, which
        # is faster than inhibition can track. So the decomposition is repeated on currents
        # boxcar-averaged over the count window T (cf. SOM Fig. S6, filtered currents).
        width = int(round(analysis["count_window_T_ms"] / lsm["asynchrony"]["current_sample_ms"]))
        cumulative = np.cumsum(components, axis=2)
        filtered = (cumulative[:, :, width:] - cumulative[:, :, :-width]) / width
        row.update({f"{k}_T": v for k, v in current_component_correlations(filtered).items()})
    return row
