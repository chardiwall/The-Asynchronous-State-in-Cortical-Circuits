"""Seam: build_weights -- S-Eq(1) connectivity for the recurrent binary network (Fig. 2)."""
import numpy as np

from block2.connectivity import build_weights

N = 200
P = 0.2
J = {"EE": 5.0, "EI": -10.0, "EX": 5.0, "IE": 5.0, "II": -9.0, "IX": 4.0}


def test_entries_are_only_zero_or_the_scaled_weight():
    weights = build_weights(n=N, p=P, j=J, rng=np.random.default_rng(1))

    expected = 5.0 / np.sqrt(N)
    entries = np.unique(weights["EE"])
    assert set(np.round(entries, 12)) <= {0.0, round(expected, 12)}


def test_density_converges_to_p():
    weights = build_weights(n=N, p=P, j=J, rng=np.random.default_rng(2))

    density = np.count_nonzero(weights["EX"]) / (N * N)
    assert abs(density - P) < 0.02


def test_within_population_diagonal_is_excluded():
    weights = build_weights(n=N, p=P, j=J, rng=np.random.default_rng(3))

    assert np.all(np.diag(weights["EE"]) == 0.0)
    assert np.all(np.diag(weights["II"]) == 0.0)


def test_cross_population_pairs_are_drawn_independently():
    weights = build_weights(n=N, p=P, j=J, rng=np.random.default_rng(4))

    assert not np.array_equal(weights["EI"] != 0, weights["IE"] != 0)
