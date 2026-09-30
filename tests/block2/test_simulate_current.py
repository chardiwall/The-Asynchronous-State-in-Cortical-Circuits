"""Seam: simulate_fast_current -- wraps _run_jit_current with connectivity construction and
a random E-cell subsample, the production entry point block2.full_pass_current uses per
realisation. Only E cells are recorded: the decomposition is over the components of one
cell's current, so I cells are not needed for it.
"""
import numpy as np

from block2.fast_model import simulate_fast_current

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}
COMMON = dict(n=20, p=0.5, j=J, m_x=0.3, theta=0.5, length_tau=2,
              sampling_rate=1, burn_in_tau=1, subsample_size=5)


def test_output_is_three_components_by_subsample_by_samples():
    result = simulate_fast_current(**COMMON, seed=0)

    assert result.shape == (3, 5, 2)


def test_same_seed_is_reproducible():
    assert np.array_equal(simulate_fast_current(**COMMON, seed=42),
                          simulate_fast_current(**COMMON, seed=42))


def test_subsample_is_capped_at_the_population_size():
    result = simulate_fast_current(**{**COMMON, "subsample_size": 500}, seed=0)

    assert result.shape[1] == 20
