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
