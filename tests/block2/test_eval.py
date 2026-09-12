"""Seam: population_averaged_correlation -- mean pairwise Pearson r (S-Eq 35-37's
np.corrcoef, block2's SR=1 raw sampled strings need no windowing) over many (i,j)
pairs, checked against S-Eq(28-29)'s predicted sign/scale. Expected values below
are hand-computed from the constructed sample patterns (all r=+-1 or a cancelling
mix), independent of np.corrcoef's own correctness.
"""
import numpy as np
import pytest

from block2.eval import population_averaged_correlation

SAMPLES_A = np.array([
    [0, 1, 0, 1],
    [0, 1, 0, 1],  # identical to neuron 0 -> r=+1
    [1, 0, 1, 0],  # exact opposite of neuron 0 -> r=-1
])


def test_same_population_excludes_the_diagonal():
    # off-diagonal pairs: (0,1)=1,(0,2)=-1,(1,2)=-1 (and symmetric) -> mean = -2/6
    r = population_averaged_correlation(SAMPLES_A, SAMPLES_A, exclude_matching_index=True)
    assert r == pytest.approx(-1 / 3)


def test_cross_population_uses_every_pair_with_no_exclusion():
    samples_b = np.array([
        [0, 1, 0, 1],  # identical to a0/a1 (+1), opposite of a2 (-1)
        [1, 0, 1, 0],  # opposite of a0/a1 (-1), identical to a2 (+1)
    ])
    # cross matrix: [[1,-1],[1,-1],[-1,1]] -> mean = 0
    r = population_averaged_correlation(SAMPLES_A, samples_b, exclude_matching_index=False)
    assert r == pytest.approx(0.0)
