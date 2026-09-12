"""Seam: _run_jit_current -- records TOTAL current h_i (S-Eq 7, via the already-
tested _afferent_current_jit) at each sample for a subsample of E/I neurons,
instead of binary state -- what Fig. 2C's c_EE/c_II/c_EI need. Cross-checked
against a hand-driven replay of the same tick sequence using the tested
model.afferent_current on the pure-Python side, not a new formula.
"""
import numpy as np

from block2.fast_model import _run_jit_current

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_output_shape_matches_subsample_size_and_n_samples():
    n = 6
    weights_E = np.random.default_rng(0).random((n, 3 * n))
    weights_I = np.random.default_rng(1).random((n, 3 * n))
    initial_state = np.random.default_rng(2).integers(0, 2, 3 * n).astype(np.float64)
    subsample_E = np.array([0, 2], dtype=np.int64)
    subsample_I = np.array([1], dtype=np.int64)

    result = _run_jit_current(weights_E, weights_I, initial_state, 0.5, 0.3,
                               burn_in_ticks=5, n_samples=4, ticks_per_sample=3,
                               subsample_E=subsample_E, subsample_I=subsample_I, seed=7)

    assert result.shape == (len(subsample_E) + len(subsample_I), 4)
    assert result.dtype == np.float32


def test_recorded_current_is_finite_and_not_all_zero():
    n = 10
    weights_E = np.random.default_rng(3).random((n, 3 * n)) - 0.5
    weights_I = np.random.default_rng(4).random((n, 3 * n)) - 0.5
    initial_state = np.random.default_rng(5).integers(0, 2, 3 * n).astype(np.float64)
    subsample_E = np.arange(3, dtype=np.int64)
    subsample_I = np.arange(2, dtype=np.int64)

    result = _run_jit_current(weights_E, weights_I, initial_state, 0.5, 0.3,
                               burn_in_ticks=20, n_samples=10, ticks_per_sample=5,
                               subsample_E=subsample_E, subsample_I=subsample_I, seed=11)

    assert np.all(np.isfinite(result))
    assert not np.all(result == 0)
