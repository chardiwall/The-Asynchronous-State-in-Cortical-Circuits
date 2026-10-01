"""Seam: simulate_fast_one -- a single realisation's raw (3n, n_samples) result,
the shape block2/run.py consumes directly, with no intermediate per-realisation copy
(the memory fix: no second copy into a per-population dict).
"""
import numpy as np

from lib.glauber import simulate_fast_one

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_output_shape_is_3n_by_n_samples():
    result = simulate_fast_one(n=20, p=0.5, j=J, m_x=0.3, theta=0.5,
                                length_tau=3, sampling_rate=1, burn_in_tau=1, seed=0)
    assert result.shape == (60, 3)
    assert result.dtype == np.uint8


def test_same_seed_is_reproducible():
    kwargs = dict(n=20, p=0.5, j=J, m_x=0.3, theta=0.5, length_tau=3, sampling_rate=1, burn_in_tau=1)
    first = simulate_fast_one(**kwargs, seed=42)
    second = simulate_fast_one(**kwargs, seed=42)
    assert np.array_equal(first, second)
