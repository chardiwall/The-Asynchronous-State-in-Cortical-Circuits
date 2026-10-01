"""Seam: resample_gaussian_conductances -- block 3's per-synapse peak conductances
(SOM S2.1.2, S-p.20-21)."""
import numpy as np
import pytest
from scipy.stats import truncnorm

from block3.connectivity import (
    generate_bernoulli_connectivity,
    resample_gaussian_conductances,
    sample_delays_ms,
)


def test_no_negative_conductances():
    g = resample_gaussian_conductances(
        mean=2.4, std_fraction=0.5, n=100_000, rng=np.random.default_rng(1)
    )

    assert np.all(g >= 0.0)


def test_realised_moments_match_the_papers_stated_ones():
    """The SOM specifies "Gaussian distributions of mean g and std. dev. 0.5g". Truncating a
    N(g, 0.5g) at zero shifts both moments -- the mean up by 2.76% and the spread down by
    5.9% -- so drawing from that distribution and discarding negatives does NOT give what
    the paper states. The draw is re-parameterised so the REALISED moments are the stated
    ones, which is the only way to have both Dale's law and the paper's numbers.
    """
    mean, std_fraction, n = 2.4, 0.5, 500_000

    g = resample_gaussian_conductances(mean, std_fraction, n, np.random.default_rng(2))

    standard_error = (std_fraction * mean) / np.sqrt(n)
    assert abs(g.mean() - mean) < 5 * standard_error
    assert abs(g.std() - std_fraction * mean) < 0.02 * std_fraction * mean


def test_the_correction_is_solved_not_hardcoded_for_one_spread():
    """The shift depends on the spread, so a constant tuned for 0.5 would be wrong for any
    other value. config.yaml exposes heterogeneity_std_fraction, so this must hold generally.
    """
    for std_fraction in (0.3, 0.5, 0.7):
        g = resample_gaussian_conductances(5.4, std_fraction, 400_000, np.random.default_rng(7))

        assert abs(g.mean() - 5.4) < 0.01 * 5.4, std_fraction
        assert abs(g.std() - std_fraction * 5.4) < 0.03 * std_fraction * 5.4, std_fraction


def test_a_naive_truncated_gaussian_would_fail_the_above():
    """Guards the guard: confirms the corrected moments differ measurably from what the
    uncorrected rejection sampling produced, so the tests above could actually fail.
    """
    mean, std = 2.4, 0.5 * 2.4
    naive = truncnorm((0.0 - mean) / std, np.inf, loc=mean, scale=std)

    assert naive.mean() / mean == pytest.approx(1.0276, abs=1e-3)
    assert naive.std() / mean == pytest.approx(0.4708, abs=1e-3)


def test_delays_are_within_range_and_on_the_005ms_grid():
    # Excitatory-origin delays: U[0.5, 1.5] ms, sampled at 0.05 ms resolution
    # (SOM S2.1.2, S-p.20-21).
    delays = sample_delays_ms(
        n=50_000, low_ms=0.5, high_ms=1.5, resolution_ms=0.05, rng=np.random.default_rng(3)
    )

    assert np.all(delays >= 0.5) and np.all(delays <= 1.5)
    grid_steps = delays / 0.05
    assert np.allclose(grid_steps, np.round(grid_steps))


def test_connectivity_density_converges_to_p_cross_population():
    # Connectivity is generated in numpy, before any Brian2 object exists, so
    # cpp_standalone doesn't need to read anything back at construction time.
    n_pre, n_post, p = 400, 100, 0.2
    pre_idx, post_idx = generate_bernoulli_connectivity(
        n_pre, n_post, p, rng=np.random.default_rng(6), exclude_self=False
    )

    density = len(pre_idx) / (n_pre * n_post)
    assert abs(density - p) < 0.01
    assert np.all((pre_idx >= 0) & (pre_idx < n_pre))
    assert np.all((post_idx >= 0) & (post_idx < n_post))


def test_connectivity_excludes_self_connections_within_population():
    n, p = 200, 0.2
    pre_idx, post_idx = generate_bernoulli_connectivity(
        n, n, p, rng=np.random.default_rng(7), exclude_self=True
    )

    assert not np.any(pre_idx == post_idx)
    density = len(pre_idx) / (n * (n - 1))
    assert abs(density - p) < 0.02
