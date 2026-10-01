"""Seam: current_component_correlations -- Fig. 2C's decomposition of the current
correlation into components (main text p.588: "the synaptic current to each cell consists
of an excitatory and an inhibitory component ... c can be decomposed into c_EE, c_II and
c_EI"). The defining property is that the components must SUM to the measured total
current correlation; if they don't, they are not a decomposition of anything.
"""
import numpy as np
import pytest

from block2.measure import (
    current_component_ccg,
    current_component_correlations,
    population_averaged_correlation,
)


def _components(n_cells: int, n_samples: int, seed: int) -> np.ndarray:
    """(3, n_cells, n_samples) E/I/X current components with realistic structure: a strong
    shared drive per component plus private noise, so the cross terms are genuinely non-zero
    and the E and I components anticorrelate the way the paper's do.
    """
    rng = np.random.default_rng(seed)
    shared_e = rng.normal(size=n_samples)
    shared_x = rng.normal(size=n_samples)
    e = 3.0 * shared_e + 0.5 * shared_x + rng.normal(size=(n_cells, n_samples))
    i = -2.5 * shared_e - 0.4 * shared_x + rng.normal(size=(n_cells, n_samples))
    x = 1.5 * shared_x + rng.normal(size=(n_cells, n_samples))
    return np.stack([e, i, x])


def test_the_six_components_sum_to_the_measured_total_correlation():
    components = _components(n_cells=24, n_samples=4000, seed=0)
    totals = components.sum(axis=0)

    c = current_component_correlations(components)
    measured_total = population_averaged_correlation(totals, totals, True)

    reconstructed = (c["c_EE"] + c["c_II"] + c["c_XX"]
                     + 2 * c["c_EI"] + 2 * c["c_EX"] + 2 * c["c_IX"])
    assert reconstructed == pytest.approx(measured_total, rel=1e-9)
    assert c["c_total"] == pytest.approx(measured_total, rel=1e-9)


def test_c_EI_is_negative_when_the_E_and_I_components_anticorrelate():
    # The paper's central claim: c_EE and c_II are large and positive while c_EI is large
    # and negative, which is what makes the total cancel.
    c = current_component_correlations(_components(n_cells=24, n_samples=4000, seed=1))

    assert c["c_EE"] > 0.5
    assert c["c_II"] > 0.5
    assert c["c_EI"] < -0.4


def test_independent_cells_give_near_zero_components():
    rng = np.random.default_rng(2)
    components = rng.normal(size=(3, 30, 6000))

    c = current_component_correlations(components)

    for key in ("c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total"):
        assert abs(c[key]) < 0.05, f"{key} = {c[key]}"


def test_a_common_normalisation_is_used_not_a_per_component_one():
    # Normalising each component by its OWN deviation would make c_EE a correlation
    # coefficient in its own right, bounded by 1 and NOT summing to the total. With a
    # shared drive this strongly-correlated E component would then read ~1.0; under the
    # correct common normalisation it reads its share of the total instead.
    components = _components(n_cells=16, n_samples=4000, seed=3)

    c = current_component_correlations(components)

    assert c["c_EE"] != pytest.approx(1.0, abs=0.05)


def test_the_components_sum_to_the_total_at_every_lag():
    """Fig. 2E plots the component CCGs against their total, so the decomposition has to
    hold at every lag, not just at zero.

    The catch: c_IE(lag) == c_EI(-lag), so the two coincide ONLY at zero lag. Away from it
    the total needs c_EI(lag) + c_IE(lag), not 2*c_EI(lag) -- which is why the paper's own
    Fig. 2E inset magnifies "the IE and EI CCGs" as two separate curves.
    """
    components = _components(n_cells=10, n_samples=3000, seed=4)
    max_lag = 4

    ccg = current_component_ccg(components, max_lag)

    summed = sum(ccg[f"c_{a}{b}"] for a in ("E", "I", "X") for b in ("E", "I", "X"))
    assert np.allclose(summed, ccg["c_total"], atol=1e-12)


def test_c_IE_is_c_EI_reflected_through_zero_lag():
    """CCG_IE(lag) == CCG_EI(-lag) exactly: both average the same Pearson correlation over
    the same ordered pair set with the lag negated. Fig. 2E's inset relies on this.
    """
    ccg = current_component_ccg(_components(n_cells=10, n_samples=3000, seed=5), max_lag=5)

    assert np.allclose(ccg["c_IE"], ccg["c_EI"][::-1], atol=1e-12)
    # ... and they are NOT equal to each other unreflected, or the distinction is vacuous.
    assert not np.allclose(ccg["c_IE"], ccg["c_EI"], atol=1e-6)
