"""S-Eq(1): connectivity for the recurrent binary network (Fig. 2, docs/paper/02-binary-network.md).

Each ordered population pair (post alpha <- pre beta) gets an independent Bernoulli(p) draw
per entry: J_ij = j_ab/sqrt(n) with probability p, else 0. Within-population pairs (EE, II)
have their diagonal forced to zero -- "a cell's own output is excluded from its input".

Two layouts of the same draw, in the same PAIRS order, so both consume the rng identically
and agree entry-for-entry at a shared seed (tested):

- build_weights -> six separate (n,n) matrices. The reference layout, used by the
  pure-Python model.tick/simulate oracle.
- build_weights_stacked -> the two (n,3n) matrices fast_model works in
  (weights_E=[EE|EI|EX], weights_I=[IE|II|IX]), filled one (n,n) block at a time.

The stacked builder is not an optimisation of the other, it is a memory fix: measured on
the DGX (session 2026-09-17, /usr/bin/time -v at N=8192), building six matrices and then
hstacking them pinned peak RSS at ~6.09GB regardless of how much else the process held --
glibc never returned the six now-freed matrices to the OS after `del`, so that transient
(~2x the final stacked size) stayed resident for the process's life. Writing each block
straight into its slice of the final arrays keeps only one (n,n) block transiently alive
and dropped the measured peak to ~4.21GB.
"""
import numpy as np

PAIRS = ("EE", "EI", "EX", "IE", "II", "IX")


def _block(n: int, p: float, j: dict[str, float], pair: str, rng: np.random.Generator) -> np.ndarray:
    connected = rng.random((n, n)) < p
    matrix = connected * (j[pair] / np.sqrt(n))
    if pair[0] == pair[1]:
        np.fill_diagonal(matrix, 0.0)
    return matrix


def build_weights(n: int, p: float, j: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {pair: _block(n, p, j, pair, rng) for pair in PAIRS}


def build_weights_stacked(n: int, p: float, j: dict[str, float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    weights_E = np.empty((n, 3 * n))
    weights_I = np.empty((n, 3 * n))
    for index, pair in enumerate(PAIRS):
        dest = weights_E if index < 3 else weights_I
        slot = index % 3
        dest[:, slot * n:(slot + 1) * n] = _block(n, p, j, pair, rng)
    return weights_E, weights_I


def couplings(net: dict) -> dict[str, float]:
    """config.yaml's binary_network.couplings (j_EE, j_EI, ...) re-keyed by PAIRS --
    the `j` argument every builder and simulator in this block takes.
    """
    return {pair: net["couplings"][f"j_{pair}"] for pair in PAIRS}
