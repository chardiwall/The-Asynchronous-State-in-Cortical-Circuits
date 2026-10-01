"""Seams: sample_neuron_subset, population_averaged_pairwise_correlation -- the
r_bar-over-subsample estimate (1000-E-neuron subsample, mirroring Fig. S6's own
"1000 E and 1000 I cells" convention, all pairs among the subsample)."""
import numpy as np

from analysis import spike_count_correlation
from block3.measure import population_averaged_pairwise_correlation, sample_neuron_subset


def test_subset_is_correct_size_and_within_range():
    subset = sample_neuron_subset(n_total=4000, n_sample=1000, rng=np.random.default_rng(1))

    assert len(subset) == 1000
    assert len(set(subset)) == 1000  # no duplicates
    assert np.all((subset >= 0) & (subset < 4000))


def test_subset_is_reproducible_given_the_same_seed():
    subset_a = sample_neuron_subset(4000, 1000, np.random.default_rng(7))
    subset_b = sample_neuron_subset(4000, 1000, np.random.default_rng(7))

    assert np.array_equal(subset_a, subset_b)


def test_averages_over_every_distinct_pair_not_all_ordered_pairs_or_self_pairs():
    # 4 tiny, distinct fake spike trains -- the independent source of truth is the exact
    # SET of pairs a correct implementation must use (all 6 unordered i<j pairs, excluding
    # self-pairs and not double-counting (i,j)/(j,i)), not a recomputation of the
    # correlation formula itself (spike_count_correlation is already tested/trusted).
    trains = [
        [10.0, 60.0, 110.0, 260.0],
        [15.0, 200.0, 340.0],
        [400.0, 410.0, 420.0, 430.0],
        [5.0, 105.0, 205.0, 305.0, 405.0],
    ]
    duration_ms, bin_dt_ms, window_T_ms = 500.0, 1.0, 50.0

    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    expected = np.mean([
        spike_count_correlation(trains[i], trains[j], duration_ms, bin_dt_ms, window_T_ms)
        for i, j in expected_pairs
    ])

    result = population_averaged_pairwise_correlation(trains, duration_ms, bin_dt_ms, window_T_ms)

    assert np.isclose(result, expected)


def test_a_silent_neuron_is_excluded_not_propagated_as_nan():
    # A neuron with zero spikes in the window has zero-variance windowed rate -- its
    # correlation with anything is undefined (0/0), not a real value to average in. The
    # correct semantic is to drop those pairs, not let a single silent neuron NaN out the
    # whole population's r_bar (a real possibility at full scale: an outlier near-silent
    # neuron in a 1000-neuron sample).
    trains = [
        [10.0, 60.0, 110.0, 260.0],
        [15.0, 200.0, 340.0],
        [],  # never fires
    ]
    duration_ms, bin_dt_ms, window_T_ms = 500.0, 1.0, 50.0

    expected = spike_count_correlation(trains[0], trains[1], duration_ms, bin_dt_ms, window_T_ms)

    result = population_averaged_pairwise_correlation(trains, duration_ms, bin_dt_ms, window_T_ms)

    assert np.isfinite(result)
    assert np.isclose(result, expected)
