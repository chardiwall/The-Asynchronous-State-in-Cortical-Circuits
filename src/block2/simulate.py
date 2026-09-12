"""Runs the recurrent binary network (docs/paper/02-binary-network.md) to produce
per-neuron sampled activity for the exploratory validation pass. Loops the tested
single-realisation connectivity.build_weights/model.tick over n_realisations
(ADR 0003: the vectorized realisation-batched loop is deferred to the full-pass
module, where it's actually needed for throughput).

Neither the paper nor docs/paper/06-figure-protocols.md states an initial condition
or burn-in; this session decided (researcher-confirmed): each neuron starts as an
independent Bernoulli(0.5) draw, then burn_in_tau*3n ticks are discarded before any
sample is recorded, so early-transient relaxation doesn't bias the recorded run --
proportionally more important for the short exploratory length_tau than the full
pass's 200,000 tau.
"""
from dataclasses import dataclass

import numpy as np

from block2.connectivity import build_weights
from block2.model import tick


@dataclass
class SimulationResult:
    activity: dict[str, np.ndarray]  # "E"/"I"/"X" -> (n_realisations, n, n_samples), {0,1}


def simulate(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, n_realisations: int, burn_in_tau: int, seed: int,
) -> SimulationResult:
    n_samples = length_tau * sampling_rate
    activity = {pop: np.zeros((n_realisations, n, n_samples), dtype=int) for pop in ("E", "I", "X")}

    ticks_per_tau = 3 * n
    ticks_per_sample = ticks_per_tau // sampling_rate

    for r in range(n_realisations):
        rng = np.random.default_rng(seed + r)
        weights = build_weights(n=n, p=p, j=j, rng=rng)
        state = {pop: rng.integers(0, 2, n) for pop in ("E", "I", "X")}

        for _ in range(burn_in_tau * ticks_per_tau):
            tick(state, weights, theta, m_x, rng)

        for sample_idx in range(n_samples):
            for _ in range(ticks_per_sample):
                tick(state, weights, theta, m_x, rng)
            for pop in ("E", "I", "X"):
                activity[pop][r, :, sample_idx] = state[pop]

    return SimulationResult(activity=activity)
