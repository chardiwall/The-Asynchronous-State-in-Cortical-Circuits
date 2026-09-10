"""Seam: check_increasing_trend -- a noise-robust proxy for "grows with p/r_in"
(the paper's stated qualitative result for Fig. 1B/E), reusing analysis.stationary_
correlation rather than a strict monotonicity check that a single noisy point would break.
"""
import pytest

from block1.eval import check_increasing_trend


def test_perfectly_increasing_values_pass():
    assert check_increasing_trend([0.0, 0.1, 0.2, 0.3], [0.01, 0.05, 0.09, 0.15]) is True


def test_decreasing_values_fail():
    assert check_increasing_trend([0.0, 0.1, 0.2, 0.3], [0.15, 0.09, 0.05, 0.01]) is False


def test_noisy_but_overall_increasing_values_still_pass():
    # One point out of order shouldn't fail a trend check meant to tolerate simulation
    # noise -- this is the whole reason it's a correlation threshold, not strict
    # monotonicity (import re-derivation: Spearman-like tolerance via Pearson r).
    assert check_increasing_trend([0.0, 0.1, 0.2, 0.3], [0.01, 0.06, 0.04, 0.15]) is True
