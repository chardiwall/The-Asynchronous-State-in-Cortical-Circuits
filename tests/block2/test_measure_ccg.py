"""Seam: population_averaged_ccg -- Fig. 2E's population-averaged CCG: the
already-tested population_averaged_correlation applied at each lag, not new
correlation math. Lag convention: CCG(lag) correlates a(t) with b(t+lag)
(matches S-Eq 42's s_i(t)s_j(t+tau)). Constructed so b is a known-lagged copy
of a, giving a hand-predictable peak lag independent of the code under test.
"""
import numpy as np

from block2.measure import population_averaged_ccg

RNG = np.random.default_rng(0)
A = RNG.integers(0, 2, size=(4, 50)).astype(np.float64)


def test_ccg_peaks_at_the_known_lag_where_b_trails_a():
    lag_true = 3
    b = np.zeros_like(A)
    b[:, lag_true:] = A[:, :-lag_true]  # b(t) = a(t - lag_true) -> b trails a

    ccg = population_averaged_ccg(A, b, max_lag=5, exclude_matching_index=False)

    lags = np.arange(-5, 6)
    assert lags[np.argmax(ccg)] == lag_true


def test_output_length_matches_lag_range():
    ccg = population_averaged_ccg(A, A, max_lag=4, exclude_matching_index=True)
    assert len(ccg) == 9  # -4..+4 inclusive
