"""Regression: the p*N literally-shared trains must be children of the SAME mother train as
the pooled ones (SOM S-p.20: "Each pre-synaptic train was a thinned version of the mother
train"). Drawing them as independent Poisson processes leaves them uncorrelated with the
pool, which biases the current correlation low by ~10% at r_in=0.01.
"""
import numpy as np
import pytest

from analysis import exact_current_correlation
from block1.inputs import build_pair_inputs

DURATION_MS = 400_000.0
COARSE_BIN_MS = 200.0   # >> the 5 ms mother-train jitter, so the STATIC formula applies


def _input_correlation(p: float, r_in: float, n_e: int, seed: int) -> float:
    """Correlation of the two cells' binned input-arrival counts.

    Binned at 200 ms rather than PSC-filtered on purpose. analysis.exact_current_correlation
    is a STATIC (long-run) formula, but the mother-train method jitters each spike by 5 ms,
    so a short-timescale filter measures a visibly smaller correlation -- a real, documented
    effect of the method, not a construction bug. Binning well above the jitter removes that
    confound, leaving only the thing under test: whether the shared trains carry the mother's
    correlation.
    """
    inputs = build_pair_inputs(n_e=n_e, n_i=0, p=p, r_in=r_in, rate_hz=20.0,
                               duration_ms=DURATION_MS, jitter_tau_ms=5.0,
                               rng=np.random.default_rng(seed))
    edges = np.arange(0, DURATION_MS + COARSE_BIN_MS, COARSE_BIN_MS)
    a, _ = np.histogram(inputs.e_spikes_a, bins=edges)
    b, _ = np.histogram(inputs.e_spikes_b, bins=edges)
    return float(np.corrcoef(a, b)[0, 1])


def test_shared_and_pooled_trains_come_from_one_mother():
    """With r_in > 0 the shared trains must carry the mother's correlation too. If they are
    independent Poisson the measured c falls well below the closed form, and the shortfall
    is largest at small r_in -- exactly where Fig. 1E is informative.
    """
    p, r_in, n_e = 0.2, 0.025, 250

    measured = _input_correlation(p, r_in, n_e, seed=0)
    expected = exact_current_correlation(p=p, r_in=r_in, n=n_e)

    # What the pre-fix construction gave instead, with m = p*N independent shared trains
    # and k = (1-p)*N children per cell: c = (m + k^2 r) / (N + k(k-1) r).
    m, k = p * n_e, (1 - p) * n_e
    independent_shared = (m + k * k * r_in) / (n_e + k * (k - 1) * r_in)

    assert measured == pytest.approx(expected, rel=0.02)
    assert measured > independent_shared


def test_at_r_in_zero_the_shared_fraction_alone_sets_the_correlation():
    # A control: with r_in = 0 the mother-train construction degenerates to independent
    # trains, so c must come out at p regardless of how the shared trains are drawn.
    measured = _input_correlation(p=0.2, r_in=0.0, n_e=250, seed=1)

    assert measured == pytest.approx(0.2, abs=0.02)
