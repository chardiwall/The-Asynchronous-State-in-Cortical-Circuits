"""Seam: resample_gaussian_conductances -- ADR 0004's rejection sampling for block 3's
per-synapse peak conductances (docs/paper/03-recurrent-spiking-network.md)."""
import numpy as np
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


def test_matches_zero_truncated_gaussian_moments():
    mean, std_fraction, n = 2.4, 0.5, 500_000
    std = std_fraction * mean

    g = resample_gaussian_conductances(mean, std_fraction, n, np.random.default_rng(2))

    # Independent source of truth: rejection-sampling a Gaussian conditioned on positivity
    # is mathematically the zero-truncated Gaussian (scipy.stats.truncnorm), not the
    # un-truncated Gaussian(mean, std) the resampling starts from.
    a = (0.0 - mean) / std
    expected = truncnorm(a, np.inf, loc=mean, scale=std)

    standard_error_of_mean = expected.std() / np.sqrt(n)
    assert abs(g.mean() - expected.mean()) < 5 * standard_error_of_mean
    assert abs(g.std() - expected.std()) < 0.02 * expected.std()


def test_delays_are_within_range_and_on_the_005ms_grid():
    # Excitatory-origin delays: U[0.5, 1.5] ms, sampled at 0.05 ms resolution
    # (docs/paper/03-recurrent-spiking-network.md).
    delays = sample_delays_ms(
        n=50_000, low_ms=0.5, high_ms=1.5, resolution_ms=0.05, rng=np.random.default_rng(3)
    )

    assert np.all(delays >= 0.5) and np.all(delays <= 1.5)
    grid_steps = delays / 0.05
    assert np.allclose(grid_steps, np.round(grid_steps))


def test_connectivity_density_converges_to_p_cross_population():
    # PROGRESS.md Phase 2: connectivity generated in numpy, before any Brian2 object, so
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
