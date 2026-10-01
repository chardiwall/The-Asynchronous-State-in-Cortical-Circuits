"""Seam: _run_jit_current -- at each sample records the three CURRENT COMPONENTS (E, I, X)
of a subsample of E cells, which is what Fig. 2C's and Fig. 2E's decomposition needs
(main text p.588). Components, not the total: c_EE is the correlation between the
E-components of two cells' currents.
"""
import numpy as np
import pytest

from lib.glauber import _run_jit_current


def _inputs(n, seed):
    weights_E = np.random.default_rng(seed).random((n, 3 * n)) - 0.5
    weights_I = np.random.default_rng(seed + 1).random((n, 3 * n)) - 0.5
    state = np.random.default_rng(seed + 2).integers(0, 2, 3 * n).astype(np.float64)
    return weights_E, weights_I, state


def test_output_is_three_components_per_recorded_cell():
    n = 6
    weights_E, weights_I, state = _inputs(n, 0)
    subsample_E = np.array([0, 2], dtype=np.int64)

    result = _run_jit_current(weights_E, weights_I, state, 0.5, 0.3, burn_in_ticks=5,
                               n_samples=4, ticks_per_sample=3, subsample_E=subsample_E, seed=7)

    assert result.shape == (3, 2, 4)
    assert result.dtype == np.float32


def test_recorded_components_are_finite_and_not_all_zero():
    n = 10
    weights_E, weights_I, state = _inputs(n, 3)

    result = _run_jit_current(weights_E, weights_I, state, 0.5, 0.3, burn_in_ticks=20,
                               n_samples=10, ticks_per_sample=5,
                               subsample_E=np.arange(3, dtype=np.int64), seed=11)

    assert np.all(np.isfinite(result))
    assert not np.all(result == 0)


def test_the_three_components_sum_to_the_cells_total_afferent_current():
    """The decomposition is only meaningful if the parts add up to the whole. theta is
    deliberately excluded from the components -- it is a constant offset that cancels out
    of every covariance and deviation -- so the sum is the pre-threshold current.
    """
    n = 8
    weights_E, weights_I, state = _inputs(n, 5)
    subsample_E = np.array([1, 4], dtype=np.int64)

    result = _run_jit_current(weights_E, weights_I, state, 0.5, 0.3, burn_in_ticks=0,
                               n_samples=1, ticks_per_sample=0, subsample_E=subsample_E, seed=9)

    for k, cell in enumerate(subsample_E):
        expected = float(weights_E[cell] @ state)
        # rel=1e-6: the components are accumulated in float32, the reference in float64.
        assert float(result[:, k, 0].sum()) == pytest.approx(expected, rel=1e-6)
