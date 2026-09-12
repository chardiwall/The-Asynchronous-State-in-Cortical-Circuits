"""Seams for the illustrative single-realisation panels (2D, 2G) -- cheap,
short-window recordings distinct from the (N, realisation) summary sweeps:

_run_jit_population_mean -- Fig. 2D's z-scored population activity: the
MEAN state across each whole population at each sample (not per-neuron),
computed without ever storing a per-neuron array.

_run_jit_subsample_state -- Fig. 2G's per-pair r histogram: raw binary state
for a subsample of one population (E), same subsampling idea as
_run_jit_current but recording state, not current.
"""
import numpy as np

from block2.panel_traces import (
    _run_jit_cell_components,
    _run_jit_population_mean,
    _run_jit_subsample_state,
)


def _random_weights(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random((n, 3 * n)) - 0.5, rng.random((n, 3 * n)) - 0.5


def test_population_mean_shape_and_bounds():
    n = 10
    weights_E, weights_I = _random_weights(n, 0)
    initial_state = np.random.default_rng(1).integers(0, 2, 3 * n).astype(np.float64)

    result = _run_jit_population_mean(weights_E, weights_I, initial_state, 0.5, 0.3,
                                       burn_in_ticks=10, n_samples=6, ticks_per_sample=4, seed=2)

    assert result.shape == (3, 6)  # E, I, X mean activity per sample
    assert np.all((result >= 0.0) & (result <= 1.0))


def test_subsample_state_shape_and_values_are_binary():
    n = 10
    weights_E, weights_I = _random_weights(n, 3)
    initial_state = np.random.default_rng(4).integers(0, 2, 3 * n).astype(np.float64)
    subsample = np.array([0, 2, 5], dtype=np.int64)

    result = _run_jit_subsample_state(weights_E, weights_I, initial_state, 0.5, 0.3,
                                       burn_in_ticks=10, n_samples=5, ticks_per_sample=4,
                                       subsample_E=subsample, seed=5)

    assert result.shape == (3, 5)
    assert set(np.unique(result)) <= {0, 1}


def test_cell_components_sum_to_total():
    n = 10
    weights_E, weights_I = _random_weights(n, 6)
    initial_state = np.random.default_rng(7).integers(0, 2, 3 * n).astype(np.float64)

    result = _run_jit_cell_components(weights_E, weights_I, initial_state, 0.5, 0.3,
                                       burn_in_ticks=10, n_samples=5, ticks_per_sample=4,
                                       cell_index=2, seed=8)

    assert result.shape == (4, 5)  # E, I, X, Total
    assert np.allclose(result[3], result[0] + result[1] + result[2] - 0.5)
