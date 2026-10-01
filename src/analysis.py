"""The shared measurement pipeline (07-analysis-methods.md), used identically by
blocks 1, 3 and 4 -- built once, held fixed, reused at every level L1-L4.
"""
import warnings
from dataclasses import dataclass, field

import numpy as np


def stationary_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """S-Eq(35-37): Cov(x,y) = (1/L)*sum((x-mean_x)(y-mean_y)),
    r = Cov(x,y)/sqrt(Cov(x,x)*Cov(y,y)) -- textbook Pearson correlation
    (library-first: numpy.corrcoef).
    """
    return float(np.corrcoef(x, y)[0, 1])


@dataclass
class StreamingCorrelation:
    """The same Pearson correlation stationary_correlation computes, accumulated over a
    sequence of chunks instead of one array held in memory at once -- for a series too
    long to hold in full (e.g. block1.run's L=10,000s current traces, chunked over
    time). Mathematically exact, not an approximation: Cov/Var are themselves just sums,
    so summing per-chunk partial sums and combining once at the end gives the identical
    result stationary_correlation(full concatenated x, y) would (verified in
    tests/test_analysis.py against stationary_correlation on both an unchunked array and
    a chunked-vs-unchunked comparison).
    """
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = field(default=0.0)

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.n += len(x)
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_x2 += float((x * x).sum())
        self.sum_y2 += float((y * y).sum())
        self.sum_xy += float((x * y).sum())

    def correlation(self) -> float:
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        cov_xy = self.sum_xy / self.n - mean_x * mean_y
        var_x = self.sum_x2 / self.n - mean_x * mean_x
        var_y = self.sum_y2 / self.n - mean_y * mean_y
        return float(cov_xy / np.sqrt(var_x * var_y))


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


def exact_current_correlation(p: float, r_in: float, n: int) -> float:
    """Main text ref. 15 (exact): for two sums of N variables each pairwise-correlated
    by r_in, of which N*p are literally common:
        c = [p + r_in*(N-p)] / [1 + r_in*(N-1)]
    Assumes a "static"/long-run notion of pairwise correlation, which is why it differs
    from a short-timescale (e.g. PSC-filtered) measured c under the mother-train method's
    temporal jitter.
    """
    return (p + r_in * (n - p)) / (1 + r_in * (n - 1))


def approx_meq1_current_correlation(p: float, r_in: float, n: int) -> float:
    """M-Eq(1): c ~= p + N*r_in, valid only when p ~ r_in*N << 1 (first-order expansion
    of exact_current_correlation for small r_in*N). Can exceed 1 outside that regime --
    not clipped, so a caller can detect when the approximation has broken down.
    """
    return p + n * r_in


def jitter_spike_times(
    spike_times_ms, jitter_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """A jittered surrogate train: every spike independently displaced by a uniform draw
    on [-jitter_ms, +jitter_ms] (Fig. S6: "uniform random shift in [-0.5,+0.5] s per
    spike"; Fig. 3B uses jitter = 500 ms). The slow rate envelope survives but correlation
    on timescales shorter than the jitter is destroyed -- which is what makes this the null
    the measured histogram is read against.

    The returned array keeps every spike, but a spike near either end can be displaced
    outside [0, duration): windowed_rate bins with an explicit range, and numpy.histogram
    silently discards values outside it. So the count reaching the correlation is preserved
    only in the interior. At Fig. 3B's scale that is ~500 ms of a ~200 s record (~0.25%,
    two edge windows) and does not affect the null; it would matter for a short record.

    Distinct from the jitter CORRECTION of S-Eq(38-40), which uses a Gaussian of std J=4T
    and is applied to non-stationary data; this is the surrogate-generation method, and
    the paper states its distribution as uniform.
    """
    spikes = np.asarray(spike_times_ms, dtype=np.float64)
    return spikes + rng.uniform(-jitter_ms, jitter_ms, spikes.shape)


def lagged_correlation(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """Cross-correlogram of two continuous traces: Pearson correlation of x(t) with
    y(t+lag), for lag = -max_lag .. +max_lag in samples, each computed over the window
    where both traces overlap.

    Used for Fig. 3C's membrane-potential CCGs. The paper gives no formula for those
    ("we computed cross-correlograms of the voltages", S-p.21) -- S-Eq(42)'s 1/(nu_i nu_j)
    normalisation is specific to spike trains and has no meaning for a voltage. A
    correlation coefficient is the reading consistent with Fig. 3D, which plots the CCG's
    peak HEIGHT against holding potential and compares it across conditions.

    Returns NaN at every lag for a constant trace: its variance is zero, so the
    correlation is undefined rather than zero.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    ccg = np.empty(2 * max_lag + 1)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            a, b = x[:n - lag], y[lag:]
        else:
            a, b = x[-lag:], y[:n + lag]
        if a.std() == 0 or b.std() == 0:
            ccg[k] = np.nan
        else:
            ccg[k] = np.corrcoef(a, b)[0, 1]
    return ccg


def mean_of_defined_pairs(pairwise: np.ndarray, what: str) -> float:
    """The project's single rule for undefined pairwise correlations, used by every block.

    A neuron with zero-variance activity -- never spiking, never changing state, saturated --
    has an undefined correlation with everything, and numpy.corrcoef returns NaN for its
    whole row. Averaging those in voids the entire realisation, so they are excluded.

    Excluding them is not free: it shrinks the denominator and biases the estimate toward
    the more active neurons. That is the right trade against losing the realisation
    outright, but it must not happen quietly, so the count is warned. A run whose reported
    correlation rests on a shrunken pair set should say so in its log.
    """
    pairwise = np.asarray(pairwise, dtype=np.float64)
    # isfinite, not just isnan: an infinite value would otherwise be counted as excluded
    # while nanmean still folded it into the mean.
    defined = pairwise[np.isfinite(pairwise)]
    undefined = pairwise.size - defined.size
    if undefined:
        warnings.warn(
            f"{what}: {undefined} of {pairwise.size} pairs were undefined (a neuron with "
            f"zero-variance activity) and were excluded from the mean.",
            RuntimeWarning, stacklevel=2,
        )
    if defined.size == 0:
        warnings.warn(
            f"{what}: EVERY pair was undefined, so the result is not a number. The whole "
            f"population had zero-variance activity -- check the run rather than the mean.",
            RuntimeWarning, stacklevel=2,
        )
        return float("nan")
    return float(defined.mean())
