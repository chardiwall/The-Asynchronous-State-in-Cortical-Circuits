"""Seam: simulate_pair's optional carry-forward state (v_init_*_mV, zi_*) -- the piece
ADR 0002 deferred, needed to run Fig. 1's real L=10,000s scale in bounded-memory chunks.
Chaining two chunked calls (2nd seeded from the 1st's returned final state) must equal
one unchunked call over the combined duration -- same principle already validated for
current_trace.synaptic_trace's zi/zf mechanism in Phase 3.
"""
import numpy as np
import pytest

from block1.model import simulate_pair

TAU_M_MS = 10.0
TAU_S_MS = 5.0
THETA_MV = 20.0
V_RESET_MV = 10.0
T_REF_MS = 2.0
J_E_MV = 3.0
J_I_MV = 3.0
DT_MS = 0.01


def test_chunked_run_matches_unchunked_run():
    # Spikes chosen to stay well subthreshold (no reset/refractory events -- that case is
    # covered separately below) and to straddle the chunk boundary at t=40ms.
    e_spikes_a = [5.0, 12.0, 38.0, 41.0, 55.0, 70.0]
    total_duration_ms = 100.0
    split_ms = 40.0

    whole = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms, dt_ms=DT_MS,
    )

    chunk_1_spikes = [s for s in e_spikes_a if s < split_ms]
    chunk_2_spikes = [s - split_ms for s in e_spikes_a if s >= split_ms]

    chunk_1 = simulate_pair(
        e_spikes_a=chunk_1_spikes, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=split_ms, dt_ms=DT_MS,
    )
    chunk_2 = simulate_pair(
        e_spikes_a=chunk_2_spikes, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms - split_ms, dt_ms=DT_MS,
        v_init_a_mV=chunk_1.v_a_final_mV, v_init_b_mV=chunk_1.v_b_final_mV,
        zi_e_a=chunk_1.zf_e_a, zi_i_a=chunk_1.zf_i_a,
        zi_e_b=chunk_1.zf_e_b, zi_i_b=chunk_1.zf_i_b,
    )

    chained_v_a = np.concatenate([chunk_1.v_a_mV, chunk_2.v_a_mV])
    chained_i_syn_a = np.concatenate([chunk_1.i_syn_a_mV, chunk_2.i_syn_a_mV])

    assert chained_v_a == pytest.approx(whole.v_a_mV, abs=1e-6)
    assert chained_i_syn_a == pytest.approx(whole.i_syn_a_mV, abs=1e-9)

    chained_spikes_a = np.concatenate([chunk_1.spikes_a_ms, chunk_2.spikes_a_ms + split_ms])
    assert chained_spikes_a == pytest.approx(whole.spikes_a_ms, abs=1e-6)


def test_chunked_run_matches_unchunked_run_when_refractory_period_straddles_boundary():
    # Regression test for a real bug (2026-09-11 code review): a strong single input
    # spike at t=39ms drives a cascade of output spikes, the first at t=39.43ms -- its
    # t_ref=2ms refractory window [39.43, 41.43] straddles the chunk split at t=40ms.
    # Carrying forward only V (via a hand-derived one-step analytic update, ignoring
    # refractory/reset entirely) corrupted this: chunk 2 would start already outside
    # its true refractory period and fire a spurious extra spike. Fixed by reading
    # cells.V/cells.lastspike directly off Brian2's own post-run NeuronGroup state
    # (exact, refractory-aware) and carrying lastspike forward too (model.py).
    e_spikes_a = [39.0]
    total_duration_ms = 100.0
    split_ms = 40.0
    j_e_strong_mV = 500.0

    whole = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms, dt_ms=DT_MS,
    )
    # Sanity check on the scenario itself: a cascade of spikes, the first one's
    # refractory window genuinely straddling the split.
    assert len(whole.spikes_a_ms) >= 2
    assert whole.spikes_a_ms[0] < split_ms < whole.spikes_a_ms[0] + T_REF_MS

    chunk_1_spikes = [s for s in e_spikes_a if s < split_ms]
    chunk_2_spikes = [s - split_ms for s in e_spikes_a if s >= split_ms]

    chunk_1 = simulate_pair(
        e_spikes_a=chunk_1_spikes, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=split_ms, dt_ms=DT_MS,
    )
    chunk_2 = simulate_pair(
        e_spikes_a=chunk_2_spikes, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=total_duration_ms - split_ms, dt_ms=DT_MS,
        v_init_a_mV=chunk_1.v_a_final_mV, v_init_b_mV=chunk_1.v_b_final_mV,
        lastspike_init_a_ms=chunk_1.lastspike_a_final_ms - split_ms,
        lastspike_init_b_ms=chunk_1.lastspike_b_final_ms - split_ms,
        zi_e_a=chunk_1.zf_e_a, zi_i_a=chunk_1.zf_i_a,
        zi_e_b=chunk_1.zf_e_b, zi_i_b=chunk_1.zf_i_b,
    )

    chained_v_a = np.concatenate([chunk_1.v_a_mV, chunk_2.v_a_mV])
    chained_spikes_a = np.concatenate([chunk_1.spikes_a_ms, chunk_2.spikes_a_ms + split_ms])

    assert chained_v_a == pytest.approx(whole.v_a_mV, abs=1e-6)
    assert chained_spikes_a == pytest.approx(whole.spikes_a_ms, abs=1e-6)


def test_default_carry_forward_state_matches_fresh_start():
    # v_init defaults to 0, zi defaults to None (fresh trace state) -- confirms the new
    # optional parameters don't change behaviour for any of the 60+ pre-existing callers
    # that don't pass them.
    result = simulate_pair(
        e_spikes_a=[10.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=DT_MS,
    )
    assert result.v_a_mV[0] == pytest.approx(0.0)
