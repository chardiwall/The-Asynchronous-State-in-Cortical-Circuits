"""Input generation for Block 1 (Fig. 1): the mother-train correlated-Poisson method
(SOM S-p.20) and the p+r_in combination confirmed with the researcher (2026-09-10):
p*N inputs are literally shared verbatim between the two cells; the
remaining (1-p)N are drawn from a SINGLE shared mother-train pool across both cells (and,
for the E+I condition, across both populations) -- not per-index-pair independent
mothers, which would fail to reproduce M-Eq(1)'s N*r_in amplification.
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


def shared_count(p: float, n: int) -> int:
    """floor(p*n), guarded against p*n's binary floating-point representation landing
    just below an intended integer (e.g. (11/15)*150 == 109.99999999999999 in float64,
    not 110.0) -- a real bug caught by review that silently moved one train from
    "shared" to "pooled" for specific (p, n) combinations.
    """
    return int(np.floor(p * n + 1e-9))


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
    """p*N of each population's inputs are literally shared verbatim between the two cells;
    the remaining (1-p)N differ between them. ALL of them -- shared and unshared alike --
    are children of ONE mother train spanning both populations and both cells, so every
    pair among a cell's inputs is correlated at r_in, which is what reproduces M-Eq(1)'s
    N*r_in amplification (a single pool, not per-index-pair independent mothers;
    researcher-confirmed 2026-09-10).
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1] (it's a shared fraction), got {p}")

    n_shared_e = shared_count(p, n_e)
    n_shared_i = shared_count(p, n_i)
    n_pool_e = n_e - n_shared_e
    n_pool_i = n_i - n_shared_i

    # EVERY train comes from the one mother pool, shared and pooled alike -- SOM S-p.20:
    # "Each pre-synaptic train was a thinned version of the mother train". Drawing the
    # p*N shared trains as independent Poisson processes instead (as this did until
    # 2026-09-30) leaves them uncorrelated with the pool and with each other, which biases
    # the current correlation low by ~10% at r_in=0.01 and ~5.5% at r_in=0.025 -- the
    # latter being the marked circle of Figs. 1C and 1F. The shared trains are drawn ONCE
    # and handed to both cells, which is what makes them literally shared.
    pool = mother_train_pool(
        rate_hz, r_in, n_shared_e + n_shared_i + 2 * (n_pool_e + n_pool_i),
        duration_ms, jitter_tau_ms, rng,
    )
    cut = 0
    shared_e, cut = pool[cut:cut + n_shared_e], cut + n_shared_e
    shared_i, cut = pool[cut:cut + n_shared_i], cut + n_shared_i
    pool_e_a, cut = pool[cut:cut + n_pool_e], cut + n_pool_e
    pool_e_b, cut = pool[cut:cut + n_pool_e], cut + n_pool_e
    pool_i_a, cut = pool[cut:cut + n_pool_i], cut + n_pool_i
    pool_i_b = pool[cut:cut + n_pool_i]

    return PairInputs(
        e_spikes_a=_pool(shared_e + pool_e_a),
        i_spikes_a=_pool(shared_i + pool_i_a),
        e_spikes_b=_pool(shared_e + pool_e_b),
        i_spikes_b=_pool(shared_i + pool_i_b),
    )
