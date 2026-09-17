"""Seam: build_weights_stacked -- builds weights_E/weights_I ((n,3n) each,
[EE|EI|EX] / [IE|II|IX] column layout) directly, without build_weights' six
separate (n,n) matrices ever existing simultaneously. Same S-Eq(1) math and
diagonal-exclusion rule as build_weights, checked the same way its own tests
are, plus a direct comparison against build_weights' hstacked output at a
shared seed for structural equivalence.
"""
import numpy as np

from block2.connectivity import build_weights
from block2.connectivity_stacked import build_weights_stacked

N = 200
P = 0.2
J = {"EE": 5.0, "EI": -10.0, "EX": 5.0, "IE": 5.0, "II": -9.0, "IX": 4.0}


def test_matches_build_weights_hstacked_at_the_same_seed():
    weights_E, weights_I = build_weights_stacked(n=N, p=P, j=J, rng=np.random.default_rng(1))
    reference = build_weights(n=N, p=P, j=J, rng=np.random.default_rng(1))
    expected_E = np.hstack([reference["EE"], reference["EI"], reference["EX"]])
    expected_I = np.hstack([reference["IE"], reference["II"], reference["IX"]])

    assert np.array_equal(weights_E, expected_E)
    assert np.array_equal(weights_I, expected_I)


def test_shapes_are_n_by_3n():
    weights_E, weights_I = build_weights_stacked(n=N, p=P, j=J, rng=np.random.default_rng(2))
    assert weights_E.shape == (N, 3 * N)
    assert weights_I.shape == (N, 3 * N)
