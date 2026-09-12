"""Illustrative single-realisation recordings for Fig. 2D and 2G -- short,
cheap runs distinct from full_pass's (N, realisation) summary sweeps. Reuses
the same tested tick (fast_model._tick_jit) and current (_afferent_current_jit)
primitives; only what gets recorded at each sample differs.
"""
import numpy as np
from numba import njit

from block2.connectivity import build_weights
from block2.fast_model import _tick_jit


@njit
def _run_jit_population_mean(
    weights_E: np.ndarray, weights_I: np.ndarray, initial_state: np.ndarray,
    theta: float, m_x: float, burn_in_ticks: int, n_samples: int, ticks_per_sample: int, seed: int,
) -> np.ndarray:
    """Fig. 2D: mean state across each whole population (E, I, X) at each
    sample -- never stores a per-neuron array, so N doesn't drive memory here.
    """
    n = weights_E.shape[0]
    state = initial_state.copy()
    np.random.seed(seed)

    for _ in range(burn_in_ticks):
        _tick_jit(state, weights_E, weights_I, theta, m_x, n)

    means = np.zeros((3, n_samples))
    for sample_idx in range(n_samples):
        for _ in range(ticks_per_sample):
            _tick_jit(state, weights_E, weights_I, theta, m_x, n)
        means[0, sample_idx] = state[:n].mean()
        means[1, sample_idx] = state[n:2 * n].mean()
        means[2, sample_idx] = state[2 * n:].mean()
    return means


@njit
def _run_jit_subsample_state(
    weights_E: np.ndarray, weights_I: np.ndarray, initial_state: np.ndarray,
    theta: float, m_x: float, burn_in_ticks: int, n_samples: int, ticks_per_sample: int,
    subsample_E: np.ndarray, seed: int,
) -> np.ndarray:
    """Fig. 2G: raw binary state for a subsample of E neurons at each sample --
    what the per-pair r histogram needs (not just an averaged r_EE).
    """
    n = weights_E.shape[0]
    state = initial_state.copy()
    np.random.seed(seed)

    for _ in range(burn_in_ticks):
        _tick_jit(state, weights_E, weights_I, theta, m_x, n)

    recorded = np.zeros((len(subsample_E), n_samples), dtype=np.uint8)
    for sample_idx in range(n_samples):
        for _ in range(ticks_per_sample):
            _tick_jit(state, weights_E, weights_I, theta, m_x, n)
        for k, i in enumerate(subsample_E):
            recorded[k, sample_idx] = np.uint8(state[i])
    return recorded


@njit
def _run_jit_cell_components(
    weights_E: np.ndarray, weights_I: np.ndarray, initial_state: np.ndarray,
    theta: float, m_x: float, burn_in_ticks: int, n_samples: int, ticks_per_sample: int,
    cell_index: int, seed: int,
) -> np.ndarray:
    """Fig. 2B: one E-cell's E/I/X current components plus their Total (S-Eq 7)
    at each sample -- weights_E's row layout is [EE|EI|EX], each block width n.
    """
    n = weights_E.shape[0]
    state = initial_state.copy()
    np.random.seed(seed)

    for _ in range(burn_in_ticks):
        _tick_jit(state, weights_E, weights_I, theta, m_x, n)

    row = weights_E[cell_index]
    components = np.zeros((4, n_samples))
    for sample_idx in range(n_samples):
        for _ in range(ticks_per_sample):
            _tick_jit(state, weights_E, weights_I, theta, m_x, n)
        e_component, i_component, x_component = 0.0, 0.0, 0.0
        for k in range(n):
            e_component += row[k] * state[k]
            i_component += row[n + k] * state[n + k]
            x_component += row[2 * n + k] * state[2 * n + k]
        components[0, sample_idx] = e_component
        components[1, sample_idx] = i_component
        components[2, sample_idx] = x_component
        components[3, sample_idx] = e_component + i_component + x_component - theta
    return components


def _build_run_inputs(n: int, p: float, j: dict[str, float], seed: int):
    rng = np.random.default_rng(seed)
    weights = build_weights(n=n, p=p, j=j, rng=rng)
    weights_E = np.hstack([weights["EE"], weights["EI"], weights["EX"]])
    weights_I = np.hstack([weights["IE"], weights["II"], weights["IX"]])
    initial_state = rng.integers(0, 2, 3 * n).astype(np.float64)
    return weights_E, weights_I, initial_state, rng


def population_mean_trace(n, p, j, m_x, theta, window_tau, sampling_rate, burn_in_tau, seed):
    weights_E, weights_I, initial_state, _ = _build_run_inputs(n, p, j, seed)
    ticks_per_tau = 3 * n
    return _run_jit_population_mean(
        weights_E, weights_I, initial_state, theta, m_x, burn_in_tau * ticks_per_tau,
        window_tau * sampling_rate, ticks_per_tau // sampling_rate, seed)


def subsample_state_trace(n, p, j, m_x, theta, window_tau, sampling_rate, burn_in_tau, seed, subsample_size):
    weights_E, weights_I, initial_state, rng = _build_run_inputs(n, p, j, seed)
    ticks_per_tau = 3 * n
    subsample_E = rng.choice(n, size=min(subsample_size, n), replace=False).astype(np.int64)
    return _run_jit_subsample_state(
        weights_E, weights_I, initial_state, theta, m_x, burn_in_tau * ticks_per_tau,
        window_tau * sampling_rate, ticks_per_tau // sampling_rate, subsample_E, seed)


def single_cell_components_trace(n, p, j, m_x, theta, window_tau, sampling_rate, burn_in_tau, seed, cell_index):
    weights_E, weights_I, initial_state, _ = _build_run_inputs(n, p, j, seed)
    ticks_per_tau = 3 * n
    return _run_jit_cell_components(
        weights_E, weights_I, initial_state, theta, m_x, burn_in_tau * ticks_per_tau,
        window_tau * sampling_rate, ticks_per_tau // sampling_rate, cell_index, seed)
