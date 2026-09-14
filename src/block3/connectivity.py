"""Per-synapse connectivity parameters for block 3 (docs/paper/03-recurrent-spiking-network.md):
conductance heterogeneity (ADR 0004) and conduction delays.
"""
import numpy as np


def resample_gaussian_conductances(
    mean: float, std_fraction: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """ADR 0004: g ~ Gaussian(mean, std_fraction*mean), any negative draw redrawn from the
    same Gaussian until non-negative (rejection sampling), independently per synapse --
    preserves Dale's law exactly rather than clipping/abs'ing/keeping negative synapses.
    """
    std = std_fraction * mean
    g = rng.normal(mean, std, n)
    negative = g < 0.0
    while np.any(negative):
        g[negative] = rng.normal(mean, std, np.count_nonzero(negative))
        negative = g < 0.0
    return g


def sample_delays_ms(
    n: int, low_ms: float, high_ms: float, resolution_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-synapse conduction delay, d ~ U[low_ms, high_ms], quantised to the simulation's
    dt=0.05ms grid (docs/paper/03-recurrent-spiking-network.md: "delays are an integer
    number of time steps").
    """
    n_steps_low = round(low_ms / resolution_ms)
    n_steps_high = round(high_ms / resolution_ms)
    steps = rng.integers(n_steps_low, n_steps_high + 1, n)
    return steps * resolution_ms


def generate_bernoulli_connectivity(
    n_pre: int, n_post: int, p: float, rng: np.random.Generator, exclude_self: bool
) -> tuple[np.ndarray, np.ndarray]:
    """docs/paper/03-recurrent-spiking-network.md: p_ij^ab ~ Bernoulli(p) independently per
    candidate synapse. Generated in plain numpy, before any Brian2 object exists, and returned
    as explicit (pre_index, post_index) arrays -- required for cpp_standalone compatibility:
    Brian2's own `Synapses.connect(p=...)` defers connectivity resolution into generated code,
    so Python can't read back how many synapses it created before that code has run (only
    matters for standalone; runtime mode resolves connect(p=...) eagerly, but this function is
    used identically in both so there is one code path, not two).

    `exclude_self` drops i==j pairs -- only meaningful when n_pre == n_post (a population
    connecting to itself), matching block 2's convention that a neuron doesn't synapse onto
    itself.
    """
    connected = rng.random(n_pre * n_post) < p
    if exclude_self:
        diagonal = np.arange(n_pre) * n_post + np.arange(n_pre)
        connected[diagonal] = False
    flat_index = np.nonzero(connected)[0]
    pre_index = (flat_index // n_post).astype(np.int32)
    post_index = (flat_index % n_post).astype(np.int32)
    return pre_index, post_index
