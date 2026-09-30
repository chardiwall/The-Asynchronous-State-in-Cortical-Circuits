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


SILENT = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [1, 1, 1, 1],   # never changes state -> zero variance -> every pair with it is undefined
])


def test_a_neuron_that_never_changes_state_is_excluded_not_propagated():
    """A neuron stuck at 0 or 1 has zero-variance activity, so np.corrcoef returns NaN for
    its whole row. Averaging that in voids the entire realisation. The defined pairs here
    are (0,1) and (1,0), both r = -1, so the answer is -1 rather than NaN.

    The spiking block already excludes such pairs; this pins the same rule here, which is
    what ticket #5 asked for.
    """
    r = population_averaged_correlation(SILENT, SILENT, exclude_matching_index=True)

    assert r == pytest.approx(-1.0)


def test_excluding_a_pair_is_reported_not_silent():
    """Excluding undefined pairs changes the denominator, which biases the estimate toward
    the more active neurons. That is the right trade against losing the realisation, but it
    must not happen quietly -- a run whose r_bar rests on a shrunken pair set should say so.
    """
    with pytest.warns(RuntimeWarning, match="undefined"):
        population_averaged_correlation(SILENT, SILENT, exclude_matching_index=True)


def test_no_warning_when_every_pair_is_defined():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        population_averaged_correlation(SAMPLES_A, SAMPLES_A, exclude_matching_index=True)
