"""Validation checks for the exploratory pass: observed simulation output against
docs/paper/02-binary-network.md's closed-form predictions (theory.py).
"""
import numpy as np


def population_averaged_correlation(
    samples_a: np.ndarray, samples_b: np.ndarray, exclude_matching_index: bool,
) -> float:
    """Mean pairwise Pearson r over every (i,j) with i in samples_a, j in samples_b.
    Block 2's r is the raw-sample correlation (docs/paper/02-binary-network.md), not
    the windowed-rate one in src/analysis.py -- library-first via np.corrcoef.
    `exclude_matching_index` drops i==j (the EE/II same-population case, where a
    neuron isn't its own pair); cross-population pairs (EI/EX/IX) need no exclusion.
    """
    n_a = samples_a.shape[0]
    corr = np.corrcoef(samples_a, samples_b)
    cross = corr[:n_a, n_a:]
    if exclude_matching_index:
        cross = cross[~np.eye(n_a, dtype=bool)]
    return float(cross.mean())


def population_averaged_ccg(
    samples_a: np.ndarray, samples_b: np.ndarray, max_lag: int, exclude_matching_index: bool,
) -> np.ndarray:
    """Fig. 2E: population_averaged_correlation applied at each lag from -max_lag
    to +max_lag (samples), not new correlation math. CCG(lag) correlates a(t)
    with b(t+lag), matching S-Eq(42)'s s_i(t)s_j(t+tau) convention.
    """
    n_samples = samples_a.shape[1]
    ccg = np.zeros(2 * max_lag + 1)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            a_slice, b_slice = samples_a[:, :n_samples - lag], samples_b[:, lag:]
        else:
            a_slice, b_slice = samples_a[:, -lag:], samples_b[:, :n_samples + lag]
        ccg[k] = population_averaged_correlation(a_slice, b_slice, exclude_matching_index)
    return ccg
