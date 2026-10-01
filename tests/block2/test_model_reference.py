"""Seam: simulate -- orchestrates connectivity.build_weights + model.tick into a full
run: random Bernoulli(0.5) initial state, burn_in_tau*3n ticks discarded, then
length_tau*sampling_rate samples recorded. Each realisation r uses an independent
rng derived as seed+r (documented reproducibility contract, exercised directly below).
"""
import numpy as np

from block2.connectivity import build_weights
from block2.model import tick
from block2.model import simulate

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}
COMMON = dict(n=4, p=0.5, j=J, m_x=0.3, theta=0.5)


def test_output_shape_and_binary_values():
    result = simulate(**COMMON, length_tau=2, sampling_rate=1, n_realisations=3, burn_in_tau=1, seed=0)

    for pop in ("E", "I", "X"):
        assert result.activity[pop].shape == (3, 4, 2)
        assert set(np.unique(result.activity[pop])) <= {0, 1}


def test_same_seed_is_reproducible():
    kwargs = dict(**COMMON, length_tau=2, sampling_rate=1, n_realisations=2, burn_in_tau=1)
    first = simulate(**kwargs, seed=42)
    second = simulate(**kwargs, seed=42)

    for pop in ("E", "I", "X"):
        assert np.array_equal(first.activity[pop], second.activity[pop])


def test_matches_a_manual_burn_in_and_sampling_loop_for_one_realisation():
    # Independent reference: build realisation 0's rng/weights the documented way
    # (seed+0), run burn_in_tau*3n ticks unrecorded, then record a sample every
    # 3n/sampling_rate ticks -- using the already-tested build_weights/tick directly,
    # not simulate() itself.
    seed, length_tau, sampling_rate, burn_in_tau = 7, 2, 1, 1
    n = COMMON["n"]
    rng = np.random.default_rng(seed + 0)
    weights = build_weights(n=n, p=COMMON["p"], j=J, rng=rng)
    state = {pop: rng.integers(0, 2, n) for pop in ("E", "I", "X")}

    ticks_per_tau = 3 * n
    for _ in range(burn_in_tau * ticks_per_tau):
        tick(state, weights, COMMON["theta"], COMMON["m_x"], rng)

    ticks_per_sample = ticks_per_tau // sampling_rate
    expected = {pop: np.zeros((n, length_tau * sampling_rate), dtype=int) for pop in ("E", "I", "X")}
    for sample_idx in range(length_tau * sampling_rate):
        for _ in range(ticks_per_sample):
            tick(state, weights, COMMON["theta"], COMMON["m_x"], rng)
        for pop in ("E", "I", "X"):
            expected[pop][:, sample_idx] = state[pop]

    result = simulate(**COMMON, length_tau=length_tau, sampling_rate=sampling_rate,
                       n_realisations=1, burn_in_tau=burn_in_tau, seed=seed)

    for pop in ("E", "I", "X"):
        assert np.array_equal(result.activity[pop][0], expected[pop])
