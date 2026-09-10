"""Input generation for Block 1 (Fig. 1): the mother-train correlated-Poisson method
(SOM S-p.20) and the p+r_in combination confirmed with the researcher (2026-09-10, see
PROGRESS.md): p*N inputs are literally shared verbatim between the two cells; the
remaining (1-p)N are drawn from a SINGLE shared mother-train pool across both cells (and,
for the E+I condition, across both populations) -- not per-index-pair independent
mothers, which would fail to reproduce M-Eq(1)'s N*r_in amplification (derivation in
PROGRESS.md).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PairInputs:
    e_spikes_a: np.ndarray
    i_spikes_a: np.ndarray
    e_spikes_b: np.ndarray
    i_spikes_b: np.ndarray


def _poisson_process(rate_hz: float, duration_ms: float, rng: np.random.Generator) -> np.ndarray:
    expected_count = rate_hz * duration_ms / 1000.0
    n = rng.poisson(expected_count)
    return np.sort(rng.uniform(0.0, duration_ms, n))


def mother_train_pool(
    rate_hz: float,
    r_in: float,
    n_children: int,
    duration_ms: float,
    jitter_tau_ms: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if not (0.0 <= r_in <= 1.0):
        raise ValueError(f"r_in must be in [0, 1] (it's a correlation), got {r_in}")

    if r_in == 0.0:
        return [_poisson_process(rate_hz, duration_ms, rng) for _ in range(n_children)]

    mother_rate_hz = rate_hz / r_in
    mother_spikes = _poisson_process(mother_rate_hz, duration_ms, rng)

    children = []
    for _ in range(n_children):
        keep = rng.random(len(mother_spikes)) < r_in
        thinned = mother_spikes[keep]
        # Two-tailed exponential (Laplace) jitter, zero mean, scale jitter_tau_ms.
        jitter = rng.laplace(0.0, jitter_tau_ms, size=len(thinned))
        jittered = thinned + jitter
        in_range = (jittered >= 0.0) & (jittered < duration_ms)
        children.append(np.sort(jittered[in_range]))

    return children


def _pool(trains: list[np.ndarray]) -> np.ndarray:
    if not trains:
        return np.array([])
    return np.sort(np.concatenate(trains))


def build_pair_inputs(
    n_e: int,
    n_i: int,
    p: float,
    r_in: float,
    rate_hz: float,
    duration_ms: float,
    jitter_tau_ms: float,
    rng: np.random.Generator,
) -> PairInputs:
    """p*N of each population's inputs are literally shared verbatim between the two
    cells; the remaining (1-p)N are drawn from ONE shared mother-train pool spanning both
    populations and both cells (researcher-confirmed, 2026-09-10 -- see PROGRESS.md for
    the derivation showing this, not per-index-pair independent mothers, is required to
    reproduce M-Eq(1)'s N*r_in amplification).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1] (it's a shared fraction), got {p}")

    n_shared_e = int(np.floor(p * n_e))
    n_shared_i = int(np.floor(p * n_i))
    n_pool_e = n_e - n_shared_e
    n_pool_i = n_i - n_shared_i

    shared_e = [_poisson_process(rate_hz, duration_ms, rng) for _ in range(n_shared_e)]
    shared_i = [_poisson_process(rate_hz, duration_ms, rng) for _ in range(n_shared_i)]

    pool = mother_train_pool(
        rate_hz, r_in, 2 * (n_pool_e + n_pool_i), duration_ms, jitter_tau_ms, rng
    )
    pool_e_a = pool[0:n_pool_e]
    pool_e_b = pool[n_pool_e : 2 * n_pool_e]
    pool_i_a = pool[2 * n_pool_e : 2 * n_pool_e + n_pool_i]
    pool_i_b = pool[2 * n_pool_e + n_pool_i : 2 * n_pool_e + 2 * n_pool_i]

    return PairInputs(
        e_spikes_a=_pool(shared_e + pool_e_a),
        i_spikes_a=_pool(shared_i + pool_i_a),
        e_spikes_b=_pool(shared_e + pool_e_b),
        i_spikes_b=_pool(shared_i + pool_i_b),
    )
