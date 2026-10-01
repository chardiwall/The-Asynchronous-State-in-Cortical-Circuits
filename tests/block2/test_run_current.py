"""Seam: run_one_current_task -- one realisation's Fig. 2C decomposition (main text p.588):
c_EE, c_II, c_XX, c_EI, c_EX, c_IX over the E/I/X components of each cell's afferent
current, plus the total they must sum to.
"""
import pytest

from block2.run import run_one_current_task

J = {"EE": 1.0, "EI": -1.0, "EX": 1.0, "IE": 1.0, "II": -1.0, "IX": 1.0}


def test_returns_all_six_components_and_a_consistent_total():
    row = run_one_current_task(n=20, realisation=1, p=0.5, j=J, m_x=0.3, theta=0.5,
                                length_tau=40, sampling_rate=1, burn_in_tau=1, seed=0,
                                subsample_size=8)

    assert row["n"] == 20 and row["realisation"] == 1
    for key in ("c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total"):
        assert key in row
    # M-Eq(3) generalised to all three components: this identity is the whole point of
    # using a common total-current normalisation.
    assert row["c_total"] == pytest.approx(
        row["c_EE"] + row["c_II"] + row["c_XX"]
        + 2 * row["c_EI"] + 2 * row["c_EX"] + 2 * row["c_IX"], rel=1e-9)
