"""Qualitative trend checks: "does this grow with that", tolerant of simulation noise.

Generic, so it lives here rather than in a block -- the paper states several of its results
as trends ("c and r_out grow roughly linearly with p") rather than as numbers, and any block
may need to check one.
"""
import math

import numpy as np

from analysis import stationary_correlation


def check_increasing_trend(
    x_values: list[float], y_values: list[float], min_correlation: float
) -> bool:
    """True if y grows with x. A correlation threshold rather than strict monotonicity, so
    one noisy point cannot fail a real trend. Fig. 1B/E state "c and r_out grow roughly
    linearly with p/r_in"."""
    r = stationary_correlation(np.array(x_values), np.array(y_values))
    if math.isnan(r):
        raise ValueError(
            "stationary_correlation is NaN (zero-variance x_values or y_values) -- "
            "the trend check is ill-posed here, not evidence of a failing trend"
        )
    return r >= min_correlation
