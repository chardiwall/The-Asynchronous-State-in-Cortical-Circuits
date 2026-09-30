"""Glauber dynamics for the recurrent binary network (Fig. 2, docs/paper/02-binary-network.md).

S-Eq(7): h_i^a = sum_b (J^{ab} . sigma^b)_i - theta, for a recurrent population a in {E,I}
and all source populations b in {E,I,X}. Inhibitory couplings are already negative in the
weight matrices (connectivity.build_weights), so this is a plain sum, no separate subtraction.
"""
import numpy as np


def afferent_current(
    state: dict[str, np.ndarray], weights: dict[str, np.ndarray], theta: float,
    population: str, i: int,
) -> float:
    total = -theta
    for source in ("E", "I", "X"):
        total += weights[population + source][i] @ state[source]
    return total


def tick(
    state: dict[str, np.ndarray], weights: dict[str, np.ndarray], theta: float,
    m_x: float, rng: np.random.Generator,
) -> tuple[str, int, int]:
    """S-p.18 exact algorithm's single elementary step: pick one of the 3N neurons
    uniformly, apply its population's update rule, mutate `state` in place. X cells
    redraw independently from Bernoulli(m_x) regardless of their previous state
    (S-Eq 6's reduction, docs/paper/02-binary-network.md); E/I cells are set to
    Theta(h_i) via afferent_current (S-Eq 5-7).
    """
    n = len(state["E"])
    idx = rng.integers(0, 3 * n)
    if idx < n:
        population, i = "E", idx
    elif idx < 2 * n:
        population, i = "I", idx - n
    else:
        population, i = "X", idx - 2 * n

    if population == "X":
        new_value = int(rng.random() < m_x)
    else:
        # Strict `> 0`, i.e. Theta(0) = 0. This follows the SOM's algorithm prose (S-p.18:
        # "if the synaptic current was LARGER THAN the firing threshold ... set to one, and
        # otherwise ... zero") rather than S-Eq(5)'s Heaviside, whose value at 0 is a
        # convention. Not merely theoretical: h == 0 exactly requires 5a - 10b + 5c = sqrt(n),
        # which is reachable in binary floating point whenever sqrt(n) is an integer multiple
        # of 5 -- n = 100 and n = 400, both exploratory sizes.
        new_value = int(afferent_current(state, weights, theta, population, i) > 0)

    state[population][i] = new_value
    return population, i, new_value
