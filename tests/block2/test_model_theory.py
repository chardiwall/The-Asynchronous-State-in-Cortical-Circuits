"""Seam: predicted_rates -- S-Eq(18)'s closed-form population-rate prediction,
m_alpha = A_alpha * m_X, from the balance condition Sum_beta J_ab m_b = -J_aX m_X
with J_ab = j_ab * p (population-level coupling, distinct from the 1/sqrt(N)-scaled
per-synapse weight in connectivity.build_weights). Expected values below are hand
solutions of the 2x2 linear system, independent of the code under test.
"""
import pytest

from block2.model import predicted_rates


def test_this_projects_actual_parameters_predict_matched_rates():
    # J_rec=[[1,-2],[1,-1.8]], J_X=[1,0.8]; A = -J_rec^-1 @ J_X = [1,1] -> m_E=m_I=m_X
    j = {"EE": 5.0, "EI": -10.0, "EX": 5.0, "IE": 5.0, "II": -9.0, "IX": 4.0}
    rates = predicted_rates(p=0.2, j=j, m_x=0.1)

    assert rates["E"] == pytest.approx(0.1)
    assert rates["I"] == pytest.approx(0.1)
    assert rates["X"] == 0.1


def test_a_different_coupling_set_predicts_unequal_e_and_i_rates():
    # J_rec=[[2,-1],[0.5,-1.5]], J_X=[1.5,1.0]; A = -J_rec^-1 @ J_X = [-0.5,0.5]
    j = {"EE": 4.0, "EI": -2.0, "EX": 3.0, "IE": 1.0, "II": -3.0, "IX": 2.0}
    rates = predicted_rates(p=0.5, j=j, m_x=0.2)

    assert rates["E"] == pytest.approx(-0.1)
    assert rates["I"] == pytest.approx(0.1)
