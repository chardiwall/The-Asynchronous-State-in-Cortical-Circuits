"""S-Eq(1): connectivity for the recurrent binary network (Fig. 2, docs/paper/02-binary-network.md).

Each ordered population pair (post alpha <- pre beta) gets an independent Bernoulli(p) draw
per entry: J_ij = j_ab/sqrt(n) with probability p, else 0. Within-population pairs (EE, II)
have their diagonal forced to zero -- "a cell's own output is excluded from its input".
"""
import numpy as np

PAIRS = ("EE", "EI", "EX", "IE", "II", "IX")


def build_weights(n: int, p: float, j: dict[str, float], rng: np.random.Generator) -> dict[str, np.ndarray]:
    weights = {}
    for pair in PAIRS:
        connected = rng.random((n, n)) < p
        matrix = connected * (j[pair] / np.sqrt(n))
        if pair[0] == pair[1]:
            np.fill_diagonal(matrix, 0.0)
        weights[pair] = matrix
    return weights
