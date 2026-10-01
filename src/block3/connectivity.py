"""Per-synapse connectivity parameters for block 3 (SOM S2.1.2, S-p.20-21):
conductance heterogeneity and conduction delays.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _truncated_normal_parameters(std_fraction: float) -> tuple[float, float]:
    """(mean, std) of the Gaussian that, once truncated at zero, has mean 1 and standard
    deviation `std_fraction` -- both expressed as multiples of the nominal conductance.

    Two equations in two unknowns, solved numerically rather than tabulated, because
    config.yaml exposes `heterogeneity_std_fraction` and a constant tuned for 0.5 would be
    silently wrong for any other value. At the paper's 0.5 the answer is (0.9486, 0.5490).
    """
    def truncated_moments(m: float, s: float) -> tuple[float, float]:
        a = -m / s
        hazard = norm.pdf(a) / (1.0 - norm.cdf(a))
        return m + s * hazard, s * np.sqrt(1.0 + a * hazard - hazard ** 2)

    def std_for(m: float) -> float:
        return brentq(lambda s: truncated_moments(m, s)[1] - std_fraction, 1e-6, 50.0)

    mean = brentq(lambda m: truncated_moments(m, std_for(m))[0] - 1.0, 1e-3, 5.0)
    return mean, std_for(mean)


def resample_gaussian_conductances(
    mean: float, std_fraction: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-synapse peak conductance: Gaussian, truncated at zero by rejection sampling, with
    the REALISED mean equal to `mean` and the realised standard deviation to
    `std_fraction * mean` -- the moments the SOM states ("Gaussian distributions of mean
    g^ab and std. dev. 0.5 g^ab").

    We redraw negative samples rather than clipping, abs-ing, or keeping them, because
    at this spread 2.3% of draws are negative and a negative conductance on an excitatory
    synapse is an inhibitory synapse -- it would break Dale's law. What that choice first
    missed (found 2026-09-30, fixed 2026-10-01) is that conditioning on a positive draw
    SHIFTS the distribution: drawing from N(g, 0.5g) and discarding negatives gives a
    realised mean of 1.0276*g and a spread of 0.4708*g, not the stated g and 0.5g. Every
    conductance in the network was therefore ~2.8% too strong.

    The fix is to draw from a different Gaussian, chosen so the truncated result has the
    stated moments. That is the only option preserving both Dale's law and the paper's
    numbers; rescaling the accepted samples fixes the mean but leaves the spread 8% low.
    """
    mean_factor, std_factor = _truncated_normal_parameters(std_fraction)
    target_mean, target_std = mean_factor * mean, std_factor * mean

    g = rng.normal(target_mean, target_std, n)
    negative = g < 0.0
    while np.any(negative):
        g[negative] = rng.normal(target_mean, target_std, np.count_nonzero(negative))
        negative = g < 0.0
    return g


def sample_delays_ms(
    n: int, low_ms: float, high_ms: float, resolution_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-synapse conduction delay, d ~ U[low_ms, high_ms], quantised to the simulation's
    dt=0.05ms grid (SOM S-p.20-21: "delays are an integer number of time steps").
    """
    n_steps_low = round(low_ms / resolution_ms)
    n_steps_high = round(high_ms / resolution_ms)
    steps = rng.integers(n_steps_low, n_steps_high + 1, n)
    return steps * resolution_ms


def generate_bernoulli_connectivity(
    n_pre: int, n_post: int, p: float, rng: np.random.Generator, exclude_self: bool
) -> tuple[np.ndarray, np.ndarray]:
    """SOM S-p.20-21: p_ij^ab ~ Bernoulli(p) independently per candidate synapse.
    Generated in plain numpy, before any Brian2 object exists, and returned
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
