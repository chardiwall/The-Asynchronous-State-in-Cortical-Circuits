"""Seam: simulate_fast_parallel -- runs realisations across worker processes instead
of sequentially. Realisations are mutually independent (own seed=seed+r, own
connectivity draw), so parallelising must be a pure orchestration change: identical
per-realisation output whether run sequentially or across workers.
"""
import numpy as np

from block2.fast_model import simulate_fast
from block2.parallel import simulate_fast_parallel

COMMON = dict(n=20, p=0.5, j={"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0},
              m_x=0.3, theta=0.5, length_tau=3, sampling_rate=1, burn_in_tau=1, seed=5)


def test_parallel_output_matches_sequential_per_realisation():
    sequential = simulate_fast(**COMMON, n_realisations=4)
    parallel = simulate_fast_parallel(**COMMON, n_realisations=4, max_workers=2)

    for pop in ("E", "I", "X"):
        assert np.array_equal(sequential.activity[pop], parallel.activity[pop])
