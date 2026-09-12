"""Seam: simulate_fast_current -- wraps _run_jit_current with connectivity
construction and a random E/I subsample, the production-facing entry point
block2.full_pass_current uses per realisation.
"""
import numpy as np

from block2.fast_model import simulate_fast_current

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_output_shapes_match_subsample_size_and_n_samples():
    result = simulate_fast_current(n=20, p=0.5, j=J, m_x=0.3, theta=0.5,
                                    length_tau=2, sampling_rate=1, burn_in_tau=1,
                                    seed=0, subsample_size=5)

    assert result["E"].shape == (5, 2)
    assert result["I"].shape == (5, 2)


def test_same_seed_is_reproducible():
    kwargs = dict(n=20, p=0.5, j=J, m_x=0.3, theta=0.5, length_tau=2,
                  sampling_rate=1, burn_in_tau=1, subsample_size=5)
    first = simulate_fast_current(**kwargs, seed=42)
    second = simulate_fast_current(**kwargs, seed=42)

    assert np.array_equal(first["E"], second["E"])
    assert np.array_equal(first["I"], second["I"])
