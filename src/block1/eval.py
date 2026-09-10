"""Qualitative trend checks for Block 1's sweeps against the paper's stated results."""
import math

import numpy as np

from analysis import stationary_correlation


def check_increasing_trend(
    x_values: list[float], y_values: list[float], min_correlation: float
) -> bool:
    """True if y grows with x, tolerant of simulation noise -- a correlation threshold
    rather than strict monotonicity, since one noisy point shouldn't fail the check.
    Fig. 1B/E's stated result is "c and r_out grow roughly linearly with p/r_in".
    min_correlation should come from config.yaml at the call site (CLAUDE.md rule 2).
    """
    r = stationary_correlation(np.array(x_values), np.array(y_values))
    if math.isnan(r):
        raise ValueError(
            "stationary_correlation is NaN (zero-variance x_values or y_values) -- "
            "the trend check is ill-posed here, not evidence of a failing trend"
        )
    return r >= min_correlation
