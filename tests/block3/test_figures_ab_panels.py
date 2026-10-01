"""Seams for Fig. 3A/3B's data: pairwise_correlations -- the vector of per-pair r values
3B histograms (its mean is the r_bar full_pass already reports); rate_sorted_sample --
3A's "500 E and I neurons sorted by rate"; binned_population_activity -- 3A's z-scored
tracking curves at the caption's 3 ms bin.
"""
import numpy as np
import pytest

from analysis import jitter_spike_times
from block3.measure import pairwise_correlations, population_averaged_pairwise_correlation
from block3.figures_ab import binned_population_activity, rate_sorted_sample

DURATION_MS = 2000.0


def _poisson_train(rate_hz, rng, duration_ms=DURATION_MS):
    n = rng.poisson(rate_hz * duration_ms / 1000.0)
    return sorted(rng.uniform(0, duration_ms, n))


def test_pairwise_correlations_returns_one_value_per_distinct_pair():
    rng = np.random.default_rng(0)
    trains = [_poisson_train(20.0, rng) for _ in range(5)]

    r = pairwise_correlations(trains, DURATION_MS, bin_dt_ms=1.0, window_T_ms=50.0)

    assert r.shape == (10,)  # C(5,2)


def test_the_mean_of_the_vector_is_the_reported_r_bar():
    rng = np.random.default_rng(1)
    trains = [_poisson_train(20.0, rng) for _ in range(6)]
    kwargs = dict(duration_ms=DURATION_MS, bin_dt_ms=1.0, window_T_ms=50.0)

    r = pairwise_correlations(trains, **kwargs)
    r_bar = population_averaged_pairwise_correlation(trains, **kwargs)

    assert float(np.nanmean(r)) == pytest.approx(r_bar)


def test_jittering_destroys_a_correlation_the_unjittered_pair_has():
    # Fig. 3B's grey histogram is the null: two trains driven by the same events are
    # correlated, and a +/-500ms jitter must wash that out.
    rng = np.random.default_rng(2)
    shared = np.array(_poisson_train(30.0, rng))
    trains = [list(shared + rng.normal(0, 2.0, shared.size)) for _ in range(4)]
    kwargs = dict(duration_ms=DURATION_MS, bin_dt_ms=1.0, window_T_ms=50.0)

    measured = np.nanmean(pairwise_correlations(trains, **kwargs))
    surrogate = np.nanmean(pairwise_correlations(
        [list(jitter_spike_times(t, 500.0, rng)) for t in trains], **kwargs))

    assert measured > 0.5
    assert abs(surrogate) < measured / 2


def test_rate_sorted_sample_is_ordered_by_rate_and_the_requested_size():
    trains = [[1.0], [1.0, 2.0, 3.0], [], [1.0, 2.0]]  # rates 1, 3, 0, 2 spikes

    indices = rate_sorted_sample(trains, n_sample=3, rng=np.random.default_rng(0))

    assert len(indices) == 3
    counts = [len(trains[i]) for i in indices]
    assert counts == sorted(counts)


def test_rate_sorted_sample_caps_at_the_population_size():
    trains = [[1.0], [2.0]]

    indices = rate_sorted_sample(trains, n_sample=10, rng=np.random.default_rng(0))

    assert len(indices) == 2


def test_binned_population_activity_counts_spikes_per_bin_across_the_population():
    # Every spike the population emitted, flat: two in bin 0, none in bin 1, one in bin 2.
    spikes = [0.5, 1.5, 6.5]

    activity = binned_population_activity(spikes, duration_ms=9.0, bin_ms=3.0)

    assert list(activity) == [2, 0, 1]


def test_binned_population_activity_handles_an_empty_population():
    activity = binned_population_activity([], duration_ms=9.0, bin_ms=3.0)

    assert list(activity) == [0, 0, 0]


def test_the_rate_matrix_budget_is_checked_before_the_run():
    """Fig. 3B's rate matrix is 1.6GB at 200 s but 40GB at the 5000 s the supplement states.
    Discovering that after the simulation would mean losing a multi-day run that wrote
    nothing, so the check has to happen before the network is even built.
    """
    from block3.figures_ab import check_rate_matrix_fits

    config = {"spiking_network": {"burn_in_ms": 100.0, "r_bar_sample_size": 1000,
                                  "populations": {"n_excitatory": 4000},
                                  "panels": {"max_rate_matrix_gb": 4.0}},
              "analysis": {"bin_dt_ms": 1.0, "count_window_T_ms": 50.0}}

    assert check_rate_matrix_fits(config, length_s=200.0) < 4.0

    with pytest.raises(ValueError, match="max_rate_matrix_gb"):
        check_rate_matrix_fits(config, length_s=5000.0)
