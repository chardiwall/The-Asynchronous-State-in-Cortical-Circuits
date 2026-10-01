"""Seam: mother_train_pool -- correlated Poisson generation (SOM S-p.20, mother-train method).

Math: one mother Poisson train at rate/r_in; each child independently keeps each mother
spike w.p. r_in (thin), then jitters every kept spike by Laplace(0, jitter_tau_ms). Every
pair among the pooled children is correlated at r_in (not just matched index pairs) --
this is required to reproduce M-Eq(1)'s N*r_in amplification (see PROGRESS.md derivation).
"""
import numpy as np
import pytest

from block1.inputs import build_pair_inputs, mother_train_pool, shared_count

RNG_SEED = 1234


def _binned_correlation(train_a, train_b, duration_ms, t_window_ms=50.0):
    bins = np.arange(0, duration_ms + t_window_ms, t_window_ms)
    counts_a, _ = np.histogram(train_a, bins=bins)
    counts_b, _ = np.histogram(train_b, bins=bins)
    return np.corrcoef(counts_a, counts_b)[0, 1]


def test_r_in_zero_gives_independent_poisson_trains_at_target_rate():
    rng = np.random.default_rng(RNG_SEED)
    duration_ms = 100_000.0  # 100s
    rate_hz = 5.0

    children = mother_train_pool(
        rate_hz=rate_hz, r_in=0.0, n_children=5,
        duration_ms=duration_ms, jitter_tau_ms=5.0, rng=rng,
    )

    assert len(children) == 5
    expected_count = rate_hz * duration_ms / 1000.0  # 500
    for train in children:
        assert train.min() >= 0.0
        assert train.max() < duration_ms
        assert np.all(np.diff(train) >= 0)  # sorted
        assert expected_count * 0.85 < len(train) < expected_count * 1.15

    # r_in=0 must not produce identical trains (that would defeat the point of "independent").
    assert not np.array_equal(children[0], children[1])


def test_mother_train_pool_produces_measurably_more_correlation_at_higher_r_in():
    # Not a precise quantitative check against r_in (that's Phase 4/6's job, once the exact
    # r_ij(T) analysis pipeline exists) -- just confirms the mechanism actually correlates
    # children, and more so at higher r_in, using a coarse independent binned-count check.
    duration_ms = 200_000.0  # 200s
    rate_hz = 5.0
    t_window_ms = 50.0

    def pairwise_count_correlation(r_in, seed):
        rng = np.random.default_rng(seed)
        children = mother_train_pool(
            rate_hz=rate_hz, r_in=r_in, n_children=2,
            duration_ms=duration_ms, jitter_tau_ms=5.0, rng=rng,
        )
        bins = np.arange(0, duration_ms + t_window_ms, t_window_ms)
        counts_0, _ = np.histogram(children[0], bins=bins)
        counts_1, _ = np.histogram(children[1], bins=bins)
        return np.corrcoef(counts_0, counts_1)[0, 1]

    corr_low = pairwise_count_correlation(r_in=0.02, seed=RNG_SEED)
    corr_high = pairwise_count_correlation(r_in=0.3, seed=RNG_SEED)

    assert corr_low == pytest.approx(0.0, abs=0.1)
    assert corr_high > 0.1
    assert corr_high > corr_low


def test_mother_train_pool_is_reproducible_given_the_same_seed():
    children_1 = mother_train_pool(
        rate_hz=5.0, r_in=0.1, n_children=3, duration_ms=10_000.0,
        jitter_tau_ms=5.0, rng=np.random.default_rng(RNG_SEED),
    )
    children_2 = mother_train_pool(
        rate_hz=5.0, r_in=0.1, n_children=3, duration_ms=10_000.0,
        jitter_tau_ms=5.0, rng=np.random.default_rng(RNG_SEED),
    )
    for a, b in zip(children_1, children_2):
        assert np.array_equal(a, b)


def test_shared_fraction_gives_exact_common_spike_times_between_cells():
    # p*N literal shared trains -- exact-value coincidences between cell_a and cell_b are
    # essentially impossible from independent continuous-time draws (float64 Poisson/
    # uniform), so any exact match must come from the shared portion.
    rng = np.random.default_rng(RNG_SEED)
    result = build_pair_inputs(
        n_e=10, n_i=0, p=0.5, r_in=0.0, rate_hz=5.0,
        duration_ms=50_000.0, jitter_tau_ms=5.0, rng=rng,
    )
    common = np.intersect1d(result.e_spikes_a, result.e_spikes_b)
    assert len(common) > 0  # from the 5 shared trains
    assert not np.array_equal(result.e_spikes_a, result.e_spikes_b)  # pool portions differ


def test_zero_shared_fraction_gives_no_common_spike_times():
    rng = np.random.default_rng(RNG_SEED)
    result = build_pair_inputs(
        n_e=10, n_i=0, p=0.0, r_in=0.0, rate_hz=5.0,
        duration_ms=50_000.0, jitter_tau_ms=5.0, rng=rng,
    )
    assert len(np.intersect1d(result.e_spikes_a, result.e_spikes_b)) == 0


def test_e_only_condition_leaves_i_arrays_empty():
    rng = np.random.default_rng(RNG_SEED)
    result = build_pair_inputs(
        n_e=10, n_i=0, p=0.2, r_in=0.05, rate_hz=5.0,
        duration_ms=10_000.0, jitter_tau_ms=5.0, rng=rng,
    )
    assert len(result.i_spikes_a) == 0
    assert len(result.i_spikes_b) == 0


def test_e_and_i_share_one_mother_train_pool_when_r_in_positive():
    # Confirms the researcher-agreed design: one mother train feeds both E and I pools, so
    # E-I pairs (same cell) end up correlated too, not just E-E/I-I within their own
    # populations. This would NOT hold if E and I used separate independent mother trains.
    duration_ms = 200_000.0
    rng = np.random.default_rng(RNG_SEED)
    result = build_pair_inputs(
        n_e=20, n_i=20, p=0.2, r_in=0.3, rate_hz=5.0,
        duration_ms=duration_ms, jitter_tau_ms=5.0, rng=rng,
    )
    cross_corr = _binned_correlation(result.e_spikes_a, result.i_spikes_a, duration_ms)
    assert cross_corr > 0.05


def test_e_and_i_do_not_spuriously_correlate_when_r_in_zero():
    duration_ms = 200_000.0
    rng = np.random.default_rng(RNG_SEED)
    result = build_pair_inputs(
        n_e=20, n_i=20, p=0.0, r_in=0.0, rate_hz=5.0,
        duration_ms=duration_ms, jitter_tau_ms=5.0, rng=rng,
    )
    cross_corr = _binned_correlation(result.e_spikes_a, result.i_spikes_a, duration_ms)
    assert cross_corr == pytest.approx(0.0, abs=0.1)


def test_r_in_above_one_is_rejected():
    # Regression for a real bug caught by review: r_in>1 makes mother_rate_hz=rate_hz/r_in
    # drop BELOW rate_hz, so thinning at keep-probability r_in silently produces trains at
    # the WRONG (lower) marginal rate instead of erroring -- e.g. r_in=2.0 gave ~10Hz
    # trains when 20Hz was requested, with no exception anywhere in the call chain.
    with pytest.raises(ValueError):
        mother_train_pool(
            rate_hz=20.0, r_in=2.0, n_children=2, duration_ms=1000.0,
            jitter_tau_ms=5.0, rng=np.random.default_rng(RNG_SEED),
        )


def test_p_above_one_is_rejected():
    # Regression for a real bug caught by review: p>1 makes n_shared_e>n_e, silently
    # generating more literal-shared trains than n_e configures, with no error.
    with pytest.raises(ValueError):
        build_pair_inputs(
            n_e=10, n_i=0, p=1.5, r_in=0.0, rate_hz=5.0,
            duration_ms=1000.0, jitter_tau_ms=5.0, rng=np.random.default_rng(RNG_SEED),
        )


def test_shared_count_is_not_off_by_one_from_floating_point_imprecision():
    # Regression for a real bug caught by review: p*n isn't always exactly
    # representable -- e.g. (11/15)*150 evaluates to 109.99999999999999 in float64, so
    # a raw floor() gives 109 instead of the mathematically intended 110, silently
    # shifting one train from "shared" to "pooled" for specific (p, n) combinations.
    assert shared_count(11 / 15, 150) == 110  # 0.7333...*150 = 109.99999999999999
    assert shared_count(11 / 15, 300) == 220  # 0.7333...*300 = 219.99999999999997
    assert shared_count(6 / 11, 220) == 120  # 0.5454...*220 = 119.99999999999999
    assert shared_count(3 / 11, 220) == 60  # 0.2727...*220 = 59.99999999999999
    assert shared_count(0.0, 250) == 0
    assert shared_count(1.0, 250) == 250
