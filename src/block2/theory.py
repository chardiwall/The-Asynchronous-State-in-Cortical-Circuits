"""Closed-form predictions from docs/paper/02-binary-network.md's theory section,
used to validate the simulation independently of it (CLAUDE.md: semantic correctness).
"""
import numpy as np


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
