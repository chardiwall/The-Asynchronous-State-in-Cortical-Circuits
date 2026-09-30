"""Numba-JIT'd Glauber dynamics (docs/paper/02-binary-network.md) -- the production
path for block 2's full pass. Same math as model.py/simulate.py, which stay the tested
pure-Python reference implementation used at exploratory scale and as this module's
correctness oracle. Session 2026-09-11: the pure-Python loop measured ~29h/realisation
at N=8192, length_tau=200,000 (~60 days for the paper's 50 realisations) -- too slow for
the full pass; this closes that gap.

nopython mode can't take a dict of named (n,n) arrays, so state and weights are
restructured: one concatenated (3n,) state vector (E,I,X in order) and two stacked
(n,3n) weight matrices, weights_E=[EE|EI|EX], weights_I=[IE|II|IX] (built directly by
connectivity.build_weights_stacked -- see its docstring for why the stacked layout is a
memory fix, not an optimisation). Cross-checked against model.py/simulate.py
statistically, not bit-for-bit: Numba's RNG and numpy's Generator are different
algorithms.

No cache=True on these @njit functions: concurrent first-time compiles of the same
function race on Numba's shared on-disk cache file, which crashed worker processes on
the DGX. Each process compiles in-memory instead (~1-2s once per process).

Recorded activity is stored as uint8, not float64: state values are strictly
Heaviside-thresholded 0/1 (S-Eq 5-7), so this is lossless, and at N=8192,
length_tau=200,000 it is the difference between ~4.9GB and ~39GB for one realisation.
"""
import numpy as np
from numba import njit

from block2.connectivity import build_weights_stacked
from block2.simulate import _ticks_per_sample


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


@njit
def _run_jit_current(
    weights_E: np.ndarray, weights_I: np.ndarray, initial_state: np.ndarray,
    theta: float, m_x: float, burn_in_ticks: int, n_samples: int, ticks_per_sample: int,
    subsample_E: np.ndarray, seed: int,
) -> np.ndarray:
    """Same tick sequence as _run_jit, but at each sample records the three CURRENT
    COMPONENTS (E, I, X) of a subsample of E cells instead of their binary state --
    what Fig. 2C's and Fig. 2E's decomposition needs (main text p.588).

    Components, not the total: c_EE is the correlation between the E-COMPONENTS of two
    cells' currents. weights_E's row layout is [EE|EI|EX], each block n wide, so the three
    components are three slices of one row -- the same inner loop
    panel_traces._run_jit_cell_components uses for Fig. 2B.

    theta is deliberately not subtracted: it is a constant offset, so it cancels out of
    every covariance and every standard deviation in the decomposition.

    Returns (3, len(subsample_E), n_samples) in E, I, X order.
    """
    n = weights_E.shape[0]
    state = initial_state.copy()
    np.random.seed(seed)

    for _ in range(burn_in_ticks):
        _tick_jit(state, weights_E, weights_I, theta, m_x, n)

    components = np.zeros((3, len(subsample_E), n_samples), dtype=np.float32)
    for sample_idx in range(n_samples):
        for _ in range(ticks_per_sample):
            _tick_jit(state, weights_E, weights_I, theta, m_x, n)
        for k in range(len(subsample_E)):
            row = weights_E[subsample_E[k]]
            e_component, i_component, x_component = 0.0, 0.0, 0.0
            for q in range(n):
                e_component += row[q] * state[q]
                i_component += row[n + q] * state[n + q]
                x_component += row[2 * n + q] * state[2 * n + q]
            components[0, k, sample_idx] = e_component
            components[1, k, sample_idx] = i_component
            components[2, k, sample_idx] = x_component
    return components


def simulate_fast_one(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int,
) -> np.ndarray:
    """One realisation, returned as the raw (3n, n_samples) uint8 array. full_pass.py
    slices E/I/X views straight off it, so no second copy into a per-population dict is
    ever made -- that copy was real, measured memory (OOM at N=8192, docs/adr/0003).
    """
    n_samples = length_tau * sampling_rate
    ticks_per_tau = 3 * n
    ticks_per_sample = _ticks_per_sample(n, sampling_rate)
    burn_in_ticks = burn_in_tau * ticks_per_tau

    rng = np.random.default_rng(seed)
    weights_E, weights_I = build_weights_stacked(n=n, p=p, j=j, rng=rng)
    initial_state = rng.integers(0, 2, 3 * n).astype(np.float64)

    return _run_jit(weights_E, weights_I, initial_state, theta, m_x,
                     burn_in_ticks, n_samples, ticks_per_sample, seed)


def simulate_fast_current(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int, subsample_size: int,
) -> np.ndarray:
    """(3, subsample, n_samples) E/I/X current components for a random subsample of E
    cells. Only E cells are recorded: the decomposition is over the components of one
    cell's current, so I cells are not needed for it.
    """
    n_samples = length_tau * sampling_rate
    ticks_per_tau = 3 * n
    ticks_per_sample = _ticks_per_sample(n, sampling_rate)
    burn_in_ticks = burn_in_tau * ticks_per_tau

    rng = np.random.default_rng(seed)
    weights_E, weights_I = build_weights_stacked(n=n, p=p, j=j, rng=rng)
    initial_state = rng.integers(0, 2, 3 * n).astype(np.float64)
    subsample_E = rng.choice(n, size=min(subsample_size, n), replace=False).astype(np.int64)

    return _run_jit_current(weights_E, weights_I, initial_state, theta, m_x,
                             burn_in_ticks, n_samples, ticks_per_sample, subsample_E, seed)
