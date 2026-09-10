"""The shared measurement pipeline (07-analysis-methods.md), used identically by
blocks 1, 3 and 4 -- built once, held fixed, reused at every level L1-L4.
"""
import numpy as np


def stationary_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """S-Eq(35-37): Cov(x,y) = (1/L)*sum((x-mean_x)(y-mean_y)),
    r = Cov(x,y)/sqrt(Cov(x,x)*Cov(y,y)) -- textbook Pearson correlation
    (library-first: numpy.corrcoef).
    """
    return float(np.corrcoef(x, y)[0, 1])


def windowed_rate(
    spike_times_ms: list[float],
    duration_ms: float,
    bin_dt_ms: float,
    window_T_ms: float,
) -> np.ndarray:
    """S-Eq(34): n_i(t;T) = K_T(t)*s_i(t), K_T = 1/T on (t,t+T) -- a forward-looking
    sliding sum of dt-binned spike counts, normalised to spikes/s. Truncated to the
    range where a full T-wide window fits (no partial windows at the boundary); the
    resulting edge loss is negligible for L>>T (Fig. 1: T=50ms, L=10,000s).
    """
    n_bins = int(round(duration_ms / bin_dt_ms))
    bins = np.arange(n_bins + 1) * bin_dt_ms
    counts, _ = np.histogram(spike_times_ms, bins=bins)

    window_samples = int(round(window_T_ms / bin_dt_ms))
    cumsum = np.concatenate(([0], np.cumsum(counts)))
    window_sums = cumsum[window_samples:] - cumsum[:-window_samples]

    t_seconds = window_T_ms / 1000.0
    return window_sums / t_seconds


def spike_count_correlation(
    spike_times_i_ms: list[float],
    spike_times_j_ms: list[float],
    duration_ms: float,
    bin_dt_ms: float,
    window_T_ms: float,
) -> float:
    """r_out: S-Eq 34's windowed rate for each spike train, then S-Eq 35-37's
    stationary correlation between them. Fig. 1's r_out uses T=50ms (config:
    analysis.count_window_T_ms), no jitter correction (Fig. 1's rates are
    stationary -- jitter correction is for non-stationary data only, see
    07-analysis-methods.md).
    """
    n_i = windowed_rate(spike_times_i_ms, duration_ms, bin_dt_ms, window_T_ms)
    n_j = windowed_rate(spike_times_j_ms, duration_ms, bin_dt_ms, window_T_ms)
    return stationary_correlation(n_i, n_j)
