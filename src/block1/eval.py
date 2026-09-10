"""Qualitative trend checks for Block 1's sweeps against the paper's stated results."""
import numpy as np

from analysis import stationary_correlation


def check_increasing_trend(
    x_values: list[float], y_values: list[float], min_correlation: float = 0.8
) -> bool:
    """True if y grows with x, tolerant of simulation noise -- a correlation threshold
    rather than strict monotonicity, since one noisy point shouldn't fail the check.
    Fig. 1B/E's stated result is "c and r_out grow roughly linearly with p/r_in".
    """
    return stationary_correlation(np.array(x_values), np.array(y_values)) >= min_correlation
