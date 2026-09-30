"""Seams: jitter_spike_times -- the uniform +/-J surrogate Fig. 3B's grey histogram is
built from (Fig. S6: "uniform random shift in [-0.5,+0.5] s per spike"); lagged_correlation
-- the membrane-potential cross-correlogram Fig. 3C averages over pairs.
"""
import numpy as np
import pytest

from analysis import jitter_spike_times, lagged_correlation


def test_jitter_preserves_the_spike_count():
    rng = np.random.default_rng(0)
    spikes = [10.0, 20.0, 30.0, 40.0]

    jittered = jitter_spike_times(spikes, jitter_ms=5.0, rng=rng)

    assert len(jittered) == len(spikes)


def test_every_spike_moves_by_at_most_the_jitter_width():
    rng = np.random.default_rng(0)
    spikes = np.arange(100, 1000, 10.0)

    jittered = jitter_spike_times(spikes, jitter_ms=25.0, rng=rng)

    # Uniform on [-J, +J]: no displacement may exceed J, and the sample must actually
    # use both signs (a one-sided shift would be a different, biased surrogate).
    displacement = np.asarray(jittered) - spikes
    assert np.all(np.abs(displacement) <= 25.0)
    assert displacement.min() < 0 < displacement.max()


def test_jitter_is_independent_per_spike_not_one_shared_shift():
    rng = np.random.default_rng(1)
    spikes = np.arange(0, 500, 5.0)

    displacement = np.asarray(jitter_spike_times(spikes, jitter_ms=50.0, rng=rng)) - spikes

    # A single shared shift applied to the whole train would leave every displacement
    # identical -- that surrogate preserves all fine-timescale correlation, so it is the
    # wrong null. Independent draws must vary.
    assert displacement.std() > 0


def test_zero_lag_correlation_matches_pearson():
    rng = np.random.default_rng(2)
    x = rng.normal(size=500)
    y = 0.7 * x + rng.normal(size=500)

    ccg = lagged_correlation(x, y, max_lag=3)

    assert ccg.shape == (7,)
    assert ccg[3] == pytest.approx(np.corrcoef(x, y)[0, 1])


def test_a_known_shift_puts_the_peak_at_that_lag():
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    # y trails x by exactly 4 samples, so corr(x(t), y(t+4)) is the maximum.
    y = np.concatenate([np.zeros(4), x[:-4]])

    ccg = lagged_correlation(x, y, max_lag=8)

    assert np.argmax(ccg) - 8 == 4


def test_a_constant_trace_gives_no_spurious_correlation():
    # A cell held at a fixed potential has zero variance; its correlation with anything
    # is undefined, and must not come back as a real number.
    ccg = lagged_correlation(np.ones(100), np.arange(100.0), max_lag=2)

    assert np.all(np.isnan(ccg))
