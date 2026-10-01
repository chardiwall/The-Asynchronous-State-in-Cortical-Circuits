"""Block 3 evaluation helpers: population rates, the r_bar-over-subsample estimate, and
Fig. 3C-D's pair enumeration and CCG peak -- the analysis side of this block, used
by run.py, figures_ab.py and figures_cd.py (SOM S-p.20-21 and S-p.24-28).
"""
import itertools

import brian2 as b2
import numpy as np

from analysis import mean_of_defined_pairs, windowed_rate


def population_rate_hz(spike_monitor, n_neurons: int, burn_in_ms: float, duration_ms: float) -> float:
    """Mean firing rate (spikes/s/neuron) after discarding the burn-in period (config.yaml:
    spiking_network.burn_in_ms -- the paper states none; an ambiguity in SOM S-p.20-21).
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
    t_ms = t_ms[after_burn_in] - burn_in_ms
    neuron_index = neuron_index[after_burn_in]

    # Grouped by one argsort + split, not one boolean scan per neuron. The per-neuron scan
    # is O(n_neurons * n_spikes): for the X population at 200 s that is 4000 neurons over
    # ~2M spikes, about 10^10 comparisons per call. Same output, sorted the same way
    # (argsort is stable, and spike_monitor.t is already ascending).
    order = np.argsort(neuron_index, kind="stable")
    boundaries = np.searchsorted(neuron_index[order], np.arange(n_neurons + 1))
    sorted_times = t_ms[order]
    return [list(sorted_times[boundaries[n]:boundaries[n + 1]]) for n in range(n_neurons)]


def sample_neuron_subset(n_total: int, n_sample: int, rng: np.random.Generator) -> np.ndarray:
    """Researcher-confirmed: r_bar is estimated over a fixed-size random subsample of E
    neurons, mirroring Fig. S6's own "1000 E and 1000 I cells" convention --
    not all N_E pairs (too many: ~8M for N_E=4000) and not an ad hoc small sample.
    """
    return rng.choice(n_total, size=n_sample, replace=False)


def pairwise_correlations(
    spike_times_by_neuron: list[list[float]], duration_ms: float, bin_dt_ms: float,
    window_T_ms: float,
) -> np.ndarray:
    """Every distinct unordered pair's spike-count correlation r (S-Eq 34-37), as a flat
    vector -- what Fig. 3B histograms. Its mean is Fig. 3A-B's r_bar.

    Each neuron's windowed rate is computed ONCE and every pair then read off a single
    vectorized numpy.corrcoef. The naive per-pair route through
    analysis.spike_count_correlation recomputes each neuron's rate ~n-1 times: measured
    impractical at n=1000 (~500k pairs, >10 min with no result).

    A neuron that never spikes in the window has zero-variance windowed rate, so corrcoef
    returns NaN for every pair it belongs to -- undefined, not zero. Those NaNs are left
    in place here so callers can see how many pairs were undefined; the r_bar wrapper
    below drops them with nanmean rather than letting one silent neuron destroy the
    estimate.
    """
    rates = np.stack([
        windowed_rate(spikes, duration_ms, bin_dt_ms, window_T_ms)
        for spikes in spike_times_by_neuron
    ])
    correlation_matrix = np.corrcoef(rates)
    return correlation_matrix[np.triu_indices(len(spike_times_by_neuron), k=1)]


def population_averaged_pairwise_correlation(
    spike_times_by_neuron: list[list[float]], duration_ms: float, bin_dt_ms: float,
    window_T_ms: float,
) -> float:
    """r_bar: the mean over pairs of pairwise_correlations, the way Fig. 3B's r_bar is
    aggregated. Pairs involving a silent neuron are undefined, not zero, so they are
    excluded and counted by analysis.mean_of_defined_pairs -- the one rule both blocks
    follow (a real possibility at full scale, not just a test artifact).
    """
    return mean_of_defined_pairs(
        pairwise_correlations(spike_times_by_neuron, duration_ms, bin_dt_ms, window_T_ms),
        "population_averaged_pairwise_correlation")


def recorded_pairs(n_a: int, n_b: int):
    """(within-A, within-B, cross) pair index lists. A-cells are 0..n_a-1 of the recorded
    array and B-cells are n_a.., matching record_membrane_potentials' ordering.
    """
    a = range(n_a)
    b = range(n_a, n_a + n_b)
    return (list(itertools.combinations(a, 2)),
            list(itertools.combinations(b, 2)),
            [(i, j) for i in a for j in b])


def signed_peak(ccg: np.ndarray) -> float:
    """Fig. 3D's "peak height of the membrane potential CCG": the extremum of the CENTRAL
    LOBE, keeping its sign.

    Two ways to get this wrong, both of which put false points on Fig. 3D:

    - A plain `max` would report a small positive side-lobe for the EPSP-IPSP condition,
      whose central peak is large and NEGATIVE (the gold curve of Fig. 3C). Hence signed.
    - A global extremum over the whole +/-50 ms window can pick a side-lobe instead of the
      peak. Fig. 3D's published y-axis is entirely non-negative and its minimum sits near
      +0.02 at rest, so the intermediate holding potentials have small central peaks with
      comparable negative excursions elsewhere in the window -- exactly the points a global
      extremum would flip negative.

    The central lobe is the contiguous run of same-signed samples containing zero lag, so
    this needs no window-width parameter: it is "the peak" in the ordinary sense, and it
    returns the correct sign for both the positive and the negative conditions.
    """
    if np.all(np.isnan(ccg)):
        return float("nan")
    centre = len(ccg) // 2
    if not np.isfinite(ccg[centre]) or ccg[centre] == 0.0:
        return float(ccg[np.nanargmax(np.abs(ccg))])

    sign = np.sign(ccg[centre])
    lo = centre
    while lo > 0 and np.isfinite(ccg[lo - 1]) and np.sign(ccg[lo - 1]) == sign:
        lo -= 1
    hi = centre
    while hi < len(ccg) - 1 and np.isfinite(ccg[hi + 1]) and np.sign(ccg[hi + 1]) == sign:
        hi += 1
    lobe = ccg[lo:hi + 1]
    return float(lobe[np.nanargmax(np.abs(lobe))])
