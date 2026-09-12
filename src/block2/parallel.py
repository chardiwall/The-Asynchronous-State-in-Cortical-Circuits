"""Spreads fast_model.simulate_fast's realisations across worker processes (DGX's 20
cores). Realisations are mutually independent -- own seed (seed+r), own connectivity
draw -- so this is pure orchestration: no new correctness surface beyond confirming
per-realisation output is identical whether run sequentially or in parallel.

Uses the 'spawn' multiprocessing start method, not the Linux default 'fork': forking
a process that has already triggered Numba/LLVM JIT compilation (as this module's own
callers typically have, e.g. running simulate_fast first) can hand child processes
partially-initialized LLVM state and crash them (BrokenProcessPool, no OOM signature
-- observed on the DGX during this session). 'spawn' gives each worker a fresh
interpreter at the cost of slightly slower startup, which is negligible next to a
full-pass run's actual duration.
"""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from block2.connectivity import build_weights
from block2.fast_model import _run_jit
from block2.simulate import SimulationResult


def _run_one_realisation(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, burn_in_tau: int, seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = build_weights(n=n, p=p, j=j, rng=rng)
    weights_E = np.hstack([weights["EE"], weights["EI"], weights["EX"]])
    weights_I = np.hstack([weights["IE"], weights["II"], weights["IX"]])
    initial_state = rng.integers(0, 2, 3 * n).astype(np.float64)

    ticks_per_tau = 3 * n
    ticks_per_sample = ticks_per_tau // sampling_rate
    burn_in_ticks = burn_in_tau * ticks_per_tau
    n_samples = length_tau * sampling_rate

    return _run_jit(weights_E, weights_I, initial_state, theta, m_x,
                     burn_in_ticks, n_samples, ticks_per_sample, seed)


def simulate_fast_parallel(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, n_realisations: int, burn_in_tau: int, seed: int,
    max_workers: int | None = None,
) -> SimulationResult:
    n_samples = length_tau * sampling_rate
    activity = {pop: np.zeros((n_realisations, n, n_samples)) for pop in ("E", "I", "X")}

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(_run_one_realisation, n, p, j, m_x, theta,
                        length_tau, sampling_rate, burn_in_tau, seed + r)
            for r in range(n_realisations)
        ]
        for r, future in enumerate(futures):
            result = future.result()
            activity["E"][r] = result[:n]
            activity["I"][r] = result[n:2 * n]
            activity["X"][r] = result[2 * n:]

    return SimulationResult(activity=activity)
