"""Validation checks for the exploratory pass: observed simulation output against
the binary network's closed-form predictions (SOM S1, S-p.3-17).
"""
import numpy as np

from analysis import mean_of_defined_pairs


def population_averaged_correlation(
    samples_a: np.ndarray, samples_b: np.ndarray, exclude_matching_index: bool,
) -> float:
    """Mean pairwise Pearson r over every (i,j) with i in samples_a, j in samples_b.
    Block 2's r is the raw-sample correlation (SOM S2.1.1, S-p.18-19), not
    the windowed-rate one in src/analysis.py -- library-first via np.corrcoef.
    `exclude_matching_index` drops i==j (the EE/II same-population case, where a
    neuron isn't its own pair); cross-population pairs (EI/EX/IX) need no exclusion.

    Undefined pairs are excluded and counted by analysis.mean_of_defined_pairs, the one
    rule both blocks follow -- a single never-changing neuron would otherwise NaN out the
    whole realisation.
    """
    n_a = samples_a.shape[0]
    corr = np.corrcoef(samples_a, samples_b)
    cross = corr[:n_a, n_a:]
    if exclude_matching_index:
        cross = cross[~np.eye(n_a, dtype=bool)]
    return mean_of_defined_pairs(cross, "population_averaged_correlation")


def population_averaged_ccg(
    samples_a: np.ndarray, samples_b: np.ndarray, max_lag: int, exclude_matching_index: bool,
) -> np.ndarray:
    """Fig. 2E: population_averaged_correlation applied at each lag from -max_lag
    to +max_lag (samples), not new correlation math. CCG(lag) correlates a(t)
    with b(t+lag), matching S-Eq(42)'s s_i(t)s_j(t+tau) convention.
    """
    n_samples = samples_a.shape[1]
    ccg = np.zeros(2 * max_lag + 1)
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            a_slice, b_slice = samples_a[:, :n_samples - lag], samples_b[:, lag:]
        else:
            a_slice, b_slice = samples_a[:, -lag:], samples_b[:, :n_samples + lag]
        ccg[k] = population_averaged_correlation(a_slice, b_slice, exclude_matching_index)
    return ccg


COMPONENTS = ("E", "I", "X")
COMPONENT_KEYS = tuple(f"c_{a}{b}" for a in COMPONENTS for b in COMPONENTS) + ("c_total",)
_SYMMETRIC_KEYS = ("c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total")


def _decompose(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """The shared kernel of Fig. 2C and Fig. 2E: the nine component cross-terms between two
    (3, n_cells, n_samples) recordings, averaged over ordered cell pairs i != j.

    Because h_i = h_i^E + h_i^I + h_i^X, Cov(h_i, h_j) expands into nine component terms.
    Dividing every term by the SAME sd(h_i)*sd(h_j) is what makes the decomposition an
    identity; normalising each component by its own deviation would instead give six
    well-formed correlation coefficients that sum to nothing in particular. Averaging over
    ordered pairs makes c_IE = c_EI (which the paper states), so

        c = c_EE + c_II + c_XX + 2*c_EI + 2*c_EX + 2*c_IX

    holds exactly, and c_total is returned so a caller can check the sum against a plainly
    measured total-current correlation rather than trusting it.

    All NINE cross-terms are returned, plus their sum as c_total. The six-term form above
    is the zero-lag special case, where c_IE == c_EI, c_XE == c_EX and c_XI == c_IX; away
    from zero lag those pairs differ and the total needs both members of each.

    O(n_cells * n_samples), not O(n_cells^2 * n_samples): the sum over pairs of
    z_i^a . z_j^b is (sum_i z_i^a).(sum_j z_j^b) minus the i == j diagonal.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n_samples = a.shape[2]
    ca = a - a.mean(axis=2, keepdims=True)
    cb = b - b.mean(axis=2, keepdims=True)
    za = ca / ca.sum(axis=0).std(axis=1)[None, :, None]   # the common normalisation
    zb = cb / cb.sum(axis=0).std(axis=1)[None, :, None]
    sum_a, sum_b = za.sum(axis=1), zb.sum(axis=1)

    n_cells = a.shape[1]
    n_pairs = n_cells * (n_cells - 1)
    result = {}
    for i, name_i in enumerate(COMPONENTS):
        for j, name_j in enumerate(COMPONENTS):
            cross = float(sum_a[i] @ sum_b[j]) / n_samples
            diagonal = float((za[i] * zb[j]).sum()) / n_samples
            result[f"c_{name_i}{name_j}"] = (cross - diagonal) / n_pairs
    result["c_total"] = sum(result[f"c_{x}{y}"] for x in COMPONENTS for y in COMPONENTS)
    return {k: result[k] for k in COMPONENT_KEYS}


def current_component_correlations(components: np.ndarray) -> dict[str, float]:
    """Fig. 2C's decomposition (main text p.588, M-Eq 3), at zero lag.

    `components` is (3, n_cells, n_samples): each recorded cell's E, I and X afferent
    current components, in COMPONENTS order. These are components of ONE cell's current,
    not the currents of different populations of cells -- the paper's c_EE is the
    correlation between the E-COMPONENTS of two cells, cross-referenced in the text to
    Fig. 1F's red and green traces.

    Returns the six distinct components: at zero lag c_IE == c_EI (which the paper states),
    c_XE == c_EX and c_XI == c_IX, so the other three carry no extra information and

        c = c_EE + c_II + c_XX + 2*c_EI + 2*c_EX + 2*c_IX

    holds. That doubling is valid ONLY at zero lag -- see current_component_ccg.
    """
    result = _decompose(components, components)
    return {k: result[k] for k in _SYMMETRIC_KEYS}


def current_component_ccg(components: np.ndarray, max_lag: int) -> dict[str, np.ndarray]:
    """Fig. 2E: the same decomposition at each lag from -max_lag to +max_lag (in samples),
    correlating cell i's component at t with cell j's at t+lag.

    Returns all NINE cross-terms, not the six of the zero-lag form, because c_IE(lag) ==
    c_EI(-lag) rather than c_EI(lag): the two coincide only at zero lag. Summing the six
    with a factor of two would therefore be wrong everywhere except the centre, and
    c_total would not match the plainly measured lagged correlation of the total currents.
    The paper makes the same distinction -- Fig. 2E's inset magnifies "the IE and EI CCGs"
    as two separate curves, which is where the EI-Lag is visible.
    """
    # Cast once, not once per lag: _decompose would otherwise promote the float32
    # recording to float64 twice on every one of the 2*max_lag+1 iterations.
    components = np.asarray(components, dtype=np.float64)
    n_samples = components.shape[2]
    ccg = {key: np.zeros(2 * max_lag + 1) for key in COMPONENT_KEYS}
    for k, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag >= 0:
            a, b = components[:, :, :n_samples - lag], components[:, :, lag:]
        else:
            a, b = components[:, :, -lag:], components[:, :, :n_samples + lag]
        for key, value in _decompose(a, b).items():
            ccg[key][k] = value
    return ccg
