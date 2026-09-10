"""Seam: simulate_pair_chunked -- runs simulate_pair over bounded time windows, carrying
V and synaptic-trace state forward between them (see model.py's carry-forward params),
so a run at Fig. 1's full L=10,000s scale doesn't need one ~15-25GB unchunked call
(ADR 0002's deferred piece). Must produce identical output to one unchunked call.
"""
import numpy as np
import pytest

from block1.chunked import simulate_pair_chunked
from block1.model import simulate_pair

TAU_M_MS = 10.0
TAU_S_MS = 5.0
THETA_MV = 20.0
V_RESET_MV = 10.0
T_REF_MS = 2.0
J_E_MV = 3.0
J_I_MV = 3.0
DT_MS = 0.01


def test_chunked_matches_unchunked_across_multiple_chunks():
    e_spikes_a = [5.0, 12.0, 27.0, 38.0, 41.0, 55.0, 62.0, 70.0, 88.0]
    total_duration_ms = 100.0

    whole = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms, dt_ms=DT_MS,
    )

    chunked = simulate_pair_chunked(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        total_duration_ms=total_duration_ms, dt_ms=DT_MS, chunk_duration_ms=30.0,  # 4 chunks
    )

    assert chunked.v_a_mV == pytest.approx(whole.v_a_mV, abs=1e-6)
    assert chunked.i_syn_a_mV == pytest.approx(whole.i_syn_a_mV, abs=1e-9)
    assert chunked.t_ms == pytest.approx(whole.t_ms, abs=1e-9)
    assert chunked.spikes_a_ms == pytest.approx(whole.spikes_a_ms, abs=1e-6)


def test_chunk_duration_not_dividing_total_duration_still_works():
    # Last chunk is shorter than chunk_duration_ms -- must not error or drop samples.
    e_spikes_a = [10.0, 45.0, 80.0]
    total_duration_ms = 100.0

    whole = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms, dt_ms=DT_MS,
    )
    chunked = simulate_pair_chunked(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        total_duration_ms=total_duration_ms, dt_ms=DT_MS, chunk_duration_ms=35.0,  # 100/35 not integer
    )

    assert len(chunked.t_ms) == len(whole.t_ms)
    assert chunked.v_a_mV == pytest.approx(whole.v_a_mV, abs=1e-6)
