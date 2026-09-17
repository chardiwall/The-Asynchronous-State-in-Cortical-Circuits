"""S-Eq(1) connectivity, built directly into fast_model's (n,3n) stacked layout
(weights_E=[EE|EI|EX], weights_I=[IE|II|IX]) instead of connectivity.build_weights'
six separate (n,n) matrices later hstacked. Same math, same diagonal-exclusion
rule (docs/paper/02-binary-network.md) -- only the allocation shape differs.

Exists because build_weights' six-matrix-then-hstack path leaves a large,
practically-unreclaimed memory footprint after `del`: measured on the DGX
(session 2026-09-17), glibc did not return the freed six-matrix memory to the
OS, so peak RSS stayed pinned near that transient's size (~2x the final
weights_E/weights_I total) for the rest of the process's life, regardless of
`del`. Building directly into the final (n,3n) arrays keeps only one (n,n)
block transiently alive at a time instead of all six.
"""
import numpy as np

_BLOCKS_E = (("EE", 0, True), ("EI", 1, False), ("EX", 2, False))
_BLOCKS_I = (("IE", 0, False), ("II", 1, True), ("IX", 2, False))


def _fill(dest: np.ndarray, n: int, p: float, j: dict[str, float], rng: np.random.Generator, blocks) -> None:
    for pair, slot, exclude_diagonal in blocks:
        connected = rng.random((n, n)) < p
        block = connected * (j[pair] / np.sqrt(n))
        if exclude_diagonal:
            np.fill_diagonal(block, 0.0)
        dest[:, slot * n:(slot + 1) * n] = block


def build_weights_stacked(n: int, p: float, j: dict[str, float], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    weights_E = np.empty((n, 3 * n))
    weights_I = np.empty((n, 3 * n))
    _fill(weights_E, n, p, j, rng, _BLOCKS_E)
    _fill(weights_I, n, p, j, rng, _BLOCKS_I)
    return weights_E, weights_I
