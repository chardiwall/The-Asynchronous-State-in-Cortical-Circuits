"""Numba-JIT'd Glauber dynamics (docs/paper/02-binary-network.md, same math as
model.py/simulate.py -- the pure-Python versions stay the tested reference
implementation, used at exploratory scale). Session 2026-09-11: the pure-Python
loop measured ~29h/realisation at N=8192, length_tau=200,000 (~60 days for the
paper's 50 realisations) -- too slow for the full pass; this closes that gap.

nopython mode can't take a dict of named (n,n) arrays, so state and weights are
restructured: one concatenated (3n,) state vector (E,I,X in order) and two
stacked (n,3n) weight matrices, weights_E=[EE|EI|EX], weights_I=[IE|II|IX].
Cross-checked against model.py/simulate.py statistically (not bit-for-bit --
Numba's RNG and numpy's Generator are different algorithms), not exactly.

No cache=True on these @njit functions: many worker processes (block2.parallel)
compiling the same function for the first time simultaneously race on Numba's
shared on-disk cache file, which crashed worker processes on the DGX
(BrokenProcessPool, no OOM/error signature -- reproduced with 20 workers, not 2).
Each process now compiles in-memory instead (a ~1-2s one-time cost per process,
negligible against a real run's duration).

Recorded activity is stored as uint8, not float64: state values are strictly
Heaviside-thresholded 0/1 (S-Eq 5-7), so this is lossless, and it matters at
scale -- at N=8192, length_tau=200,000 the float64 array would be ~39GB for a
single realisation (an 8x-oversized allocation that silently blocked real
concurrency on the DGX's Slurm nodes: each task's implicit memory footprint
left room for only 2-3 concurrent N=8192 tasks per 118GB node, found while
testing block2.full_pass there). uint8 brings that to ~4.9GB.
"""
import numpy as np
from numba import njit

from block2.connectivity import build_weights
from block2.simulate import SimulationResult


@njit
def _afferent_current_jit(state: np.ndarray, weights_row: np.ndarray, theta: float) -> float:
    # Manual loop, not np.dot: ~1.5x faster measured at N=8192 scale inside nopython
    # mode (np.dot doesn't hit an optimized BLAS path here the way numpy's does).
    total = 0.0
    for k in range(state.shape[0]):
        total += weights_row[k] * state[k]
    return total - theta


@njit
def _tick_jit(state: np.ndarray, weights_E: np.ndarray, weights_I: np.ndarray, theta: float, m_x: float, n: int) -> None:
    idx = np.random.randint(0, 3 * n)
    if idx < n:
        state[idx] = 1.0 if _afferent_current_jit(state, weights_E[idx], theta) > 0 else 0.0
    elif idx < 2 * n:
        i = idx - n
        state[n + i] = 1.0 if _afferent_current_jit(state, weights_I[i], theta) > 0 else 0.0
    else:
        i = idx - 2 * n
        state[2 * n + i] = 1.0 if np.random.random() < m_x else 0.0


@njit
def _run_jit(
    weights_E: np.ndarray, weights_I: np.ndarray, initial_state: np.ndarray,
    theta: float, m_x: float, burn_in_ticks: int, n_samples: int, ticks_per_sample: int, seed: int,
) -> np.ndarray:
    n = weights_E.shape[0]
    state = initial_state.copy()
    np.random.seed(seed)

    for _ in range(burn_in_ticks):
        _tick_jit(state, weights_E, weights_I, theta, m_x, n)

    activity = np.zeros((3 * n, n_samples), dtype=np.uint8)
    for sample_idx in range(n_samples):
        for _ in range(ticks_per_sample):
            _tick_jit(state, weights_E, weights_I, theta, m_x, n)
        activity[:, sample_idx] = state.astype(np.uint8)
    return activity


def simulate_fast(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, n_realisations: int, burn_in_tau: int, seed: int,
) -> SimulationResult:
    n_samples = length_tau * sampling_rate
    ticks_per_tau = 3 * n
    ticks_per_sample = ticks_per_tau // sampling_rate
    burn_in_ticks = burn_in_tau * ticks_per_tau

    activity = {pop: np.zeros((n_realisations, n, n_samples), dtype=np.uint8) for pop in ("E", "I", "X")}
    for r in range(n_realisations):
        rng = np.random.default_rng(seed + r)
        weights = build_weights(n=n, p=p, j=j, rng=rng)
        weights_E = np.hstack([weights["EE"], weights["EI"], weights["EX"]])
        weights_I = np.hstack([weights["IE"], weights["II"], weights["IX"]])
        initial_state = rng.integers(0, 2, 3 * n).astype(np.float64)

        result = _run_jit(weights_E, weights_I, initial_state, theta, m_x,
                           burn_in_ticks, n_samples, ticks_per_sample, seed + r)
        activity["E"][r] = result[:n]
        activity["I"][r] = result[n:2 * n]
        activity["X"][r] = result[2 * n:]

    return SimulationResult(activity=activity)
