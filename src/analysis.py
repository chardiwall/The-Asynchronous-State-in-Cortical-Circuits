"""The shared measurement pipeline (07-analysis-methods.md), used identically by
blocks 1, 3 and 4 -- built once, held fixed, reused at every level L1-L4.
"""
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
    long to hold in full (e.g. block1.full_pass's L=10,000s current traces, chunked over
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
    Assumes a "static"/long-run notion of pairwise correlation -- see PROGRESS.md
    Phase 8 for why this differs from a short-timescale (e.g. PSC-filtered) measured c
    under the mother-train method's temporal jitter.
    """
    return (p + r_in * (n - p)) / (1 + r_in * (n - 1))


def approx_meq1_current_correlation(p: float, r_in: float, n: int) -> float:
    """M-Eq(1): c ~= p + N*r_in, valid only when p ~ r_in*N << 1 (first-order expansion
    of exact_current_correlation for small r_in*N). Can exceed 1 outside that regime --
    not clipped, so a caller can detect when the approximation has broken down.
    """
    return p + n * r_in
