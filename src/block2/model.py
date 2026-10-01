"""Block 2's binary network: the paper's own equations, and the reference implementation
they are checked against.

**Dynamics** -- S-Eq(7): h_i^a = sum_b (J^{ab} . sigma^b)_i - theta, for a recurrent
population a in {E,I} over all source populations b in {E,I,X}. Inhibitory couplings are
already negative in the weight matrices, so this is a plain sum with no separate
subtraction. S-p.18's exact algorithm then picks one of the 3N neurons uniformly and
applies its population's update rule.

**Closed form** -- S-Eq(18): the balance condition sum_b J_ab m_b = -J_aX m_X gives
m_a = A_a m_X, which predicts the population rates independently of any simulation. This is
what the exploratory pass validates against.

**Reference simulation** -- the pure-Python loop below is slow and is not what production
uses (that is lib.glauber's JIT'd version). It is the correctness oracle: it is the literal
transcription of the equations above, so a statistical agreement between it and the fast
path is evidence the fast path computes the paper's maths. Deleting it would leave the JIT
code with nothing to be checked against.

Neither the paper nor its figure protocols state an initial condition or a burn-in. This
project uses an independent Bernoulli(0.5) draw per neuron, then discards burn_in_tau*3n
ticks before recording -- proportionally more important at the short exploratory length than
at the full pass's 200,000 tau.
"""
from dataclasses import dataclass

import numpy as np

from block2.connectivity import build_weights
from lib.glauber import ticks_between_samples


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


def predicted_rates(p: float, j: dict[str, float], m_x: float) -> dict[str, float]:
    """S-Eq(18): balance Sum_beta J_ab m_b = -J_aX m_X (a,b in {E,I}) gives
    m_a = A_a * m_X, A = -J_rec^-1 @ J_X. J_ab = j_ab*p is the population-level
    effective coupling -- distinct from the 1/sqrt(N)-scaled per-synapse weight
    used by connectivity.build_weights.
    """
    j_pop = {k: v * p for k, v in j.items()}
    j_rec = np.array([[j_pop["EE"], j_pop["EI"]], [j_pop["IE"], j_pop["II"]]])
    j_x = np.array([j_pop["EX"], j_pop["IX"]])

    a = -np.linalg.solve(j_rec, j_x)
    m_e, m_i = a * m_x
    return {"E": m_e, "I": m_i, "X": m_x}


@dataclass
class SimulationResult:
    activity: dict[str, np.ndarray]  # "E"/"I"/"X" -> (n_realisations, n, n_samples), {0,1}


def simulate(
    n: int, p: float, j: dict[str, float], m_x: float, theta: float,
    length_tau: int, sampling_rate: int, n_realisations: int, burn_in_tau: int, seed: int,
) -> SimulationResult:
    n_samples = length_tau * sampling_rate
    activity = {pop: np.zeros((n_realisations, n, n_samples), dtype=int) for pop in ("E", "I", "X")}

    ticks_per_tau = 3 * n
    ticks_per_sample = ticks_between_samples(n, sampling_rate)

    for r in range(n_realisations):
        rng = np.random.default_rng(seed + r)
        weights = build_weights(n=n, p=p, j=j, rng=rng)
        state = {pop: rng.integers(0, 2, n) for pop in ("E", "I", "X")}

        for _ in range(burn_in_tau * ticks_per_tau):
            tick(state, weights, theta, m_x, rng)

        for sample_idx in range(n_samples):
            for _ in range(ticks_per_sample):
                tick(state, weights, theta, m_x, rng)
            for pop in ("E", "I", "X"):
                activity[pop][r, :, sample_idx] = state[pop]

    return SimulationResult(activity=activity)
