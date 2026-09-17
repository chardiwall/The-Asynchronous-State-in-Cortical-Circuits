"""Block 3 evaluation helpers: population rates and the r_bar-over-subsample estimate used
by both exploratory_pass.py and full_pass.py (docs/paper/03-recurrent-spiking-network.md,
07-analysis-methods.md).
"""
import brian2 as b2
import numpy as np

from analysis import windowed_rate


def population_rate_hz(spike_monitor, n_neurons: int, burn_in_ms: float, duration_ms: float) -> float:
    """Mean firing rate (spikes/s/neuron) after discarding the burn-in period (config.yaml:
    spiking_network.burn_in_ms -- the paper doesn't state one, see docs/paper/03 Ambiguity).
    """
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    n_spikes_after_burn_in = int(np.sum(t_ms >= burn_in_ms))
    recorded_s = (duration_ms - burn_in_ms) / 1000.0
    return n_spikes_after_burn_in / (n_neurons * recorded_s)


def spike_times_by_neuron(spike_monitor, n_neurons: int, burn_in_ms: float) -> list[list[float]]:
    """Per-neuron spike times (ms, re-zeroed to the end of burn-in) from a Brian2
    SpikeMonitor -- the shape src/analysis.py's spike_count_correlation expects.
    """
    t_ms = np.asarray(spike_monitor.t / b2.ms)
    neuron_index = np.asarray(spike_monitor.i)
    after_burn_in = t_ms >= burn_in_ms
    return [
        list(t_ms[after_burn_in & (neuron_index == n)] - burn_in_ms) for n in range(n_neurons)
    ]


def sample_neuron_subset(n_total: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """PROGRESS.md (researcher-confirmed): r_bar is estimated over a fixed-size random
    subsample of E neurons, mirroring Fig. S6's own "1000 E and 1000 I cells" convention --
    not all N_E pairs (too many: ~8M for N_E=4000) and not an ad hoc small sample.
    """
    return rng.choice(n_total, size=n_sample, replace=False)


def population_averaged_pairwise_correlation(
    spike_times_by_neuron: list[list[float]], duration_ms: float, bin_dt_ms: float,
    window_T_ms: float,
) -> float:
    """r_bar: S-Eq(35-37)'s spike-count correlation, averaged over every distinct unordered
    pair among the given spike trains (docs/paper/07-analysis-methods.md's r_ij, aggregated
    the way Fig. 3B's r_bar is -- a mean over pairs, not a single-pair value).

    Computes each neuron's windowed rate once (not once per pair -- a naive per-pair
    implementation via analysis.spike_count_correlation recomputes every neuron's rate
    ~n-1 times, measured impractically slow at n=1000/~500k pairs: >10min with no result)
    then gets every pairwise Pearson correlation in one vectorized numpy.corrcoef call.
    """
    rates = np.stack([
        windowed_rate(spikes, duration_ms, bin_dt_ms, window_T_ms)
        for spikes in spike_times_by_neuron
    ])
    correlation_matrix = np.corrcoef(rates)
    upper_triangle = correlation_matrix[np.triu_indices(len(spike_times_by_neuron), k=1)]
    # A neuron with zero spikes in the window has zero-variance windowed rate -- its
    # correlation with anything is undefined (0/0 -> NaN from corrcoef), not a real value.
    # nanmean drops those pairs rather than letting one silent neuron NaN out the whole
    # population's r_bar -- a real possibility at full scale, not just a test artifact.
    return float(np.nanmean(upper_triangle))
