"""Seam: simulate_pairs_batch -- N independent postsynaptic pairs simulated in ONE
Brian2 NeuronGroup(2N) instead of N separate simulate_pair calls. The N points don't
couple to each other (purely feedforward, per-point pooled inputs) so batching must be
exact: point k's results must be bit-identical to what simulate_pair(k's own inputs)
alone would produce, for every k, regardless of what the other N-1 points' inputs are.
"""
import numpy as np
import pytest

from lib.brian_batch import simulate_pairs_batch
from lib.chunking import simulate_pairs_batch_chunked
from block1.model import simulate_pair

TAU_M_MS = 10.0
TAU_S_MS = 5.0
THETA_MV = 20.0
V_RESET_MV = 10.0
T_REF_MS = 2.0
J_E_MV = 3.0
J_I_MV = 3.0
DT_MS = 0.01


def test_batch_of_one_matches_simulate_pair():
    e_spikes_a = [5.0, 12.0, 27.0, 38.0, 55.0]
    i_spikes_b = [10.0, 30.0]
    duration_ms = 100.0

    solo = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=i_spikes_b,
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )
    batch = simulate_pairs_batch(
        e_spikes_a=[e_spikes_a], i_spikes_a=[[]], e_spikes_b=[[]], i_spikes_b=[i_spikes_b],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )

    assert batch.v_a_mV[:, 0] == pytest.approx(solo.v_a_mV, abs=1e-9)
    assert batch.v_b_mV[:, 0] == pytest.approx(solo.v_b_mV, abs=1e-9)
    assert batch.i_syn_a_mV[:, 0] == pytest.approx(solo.i_syn_a_mV, abs=1e-9)
    assert batch.spikes_a_ms[0] == pytest.approx(solo.spikes_a_ms, abs=1e-9)
    assert batch.spikes_b_ms[0] == pytest.approx(solo.spikes_b_ms, abs=1e-9)
    assert batch.v_a_final_mV[0] == pytest.approx(solo.v_a_final_mV, abs=1e-9)
    assert batch.lastspike_a_final_ms[0] == pytest.approx(solo.lastspike_a_final_ms, abs=1e-9)


def test_batch_points_are_independent():
    # Point 0: a strong single spike, driving cell A well above threshold (a real
    # spike/reset/refractory cascade). Point 1: no input at all (silent). Each point's
    # results must match what simulate_pair would give for that point ALONE -- no
    # leakage from point 0's spiking into point 1's silence, or vice versa.
    duration_ms = 100.0
    point0_e_spikes_a = [39.0]
    j_e_strong_mV = 500.0

    solo0 = simulate_pair(
        e_spikes_a=point0_e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )
    solo1 = simulate_pair(
        e_spikes_a=[], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )

    batch = simulate_pairs_batch(
        e_spikes_a=[point0_e_spikes_a, []], i_spikes_a=[[], []],
        e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )

    assert len(solo0.spikes_a_ms) >= 2  # sanity: point 0 really does spike repeatedly
    assert batch.spikes_a_ms[0] == pytest.approx(solo0.spikes_a_ms, abs=1e-9)
    assert batch.v_a_mV[:, 0] == pytest.approx(solo0.v_a_mV, abs=1e-9)
    assert len(batch.spikes_a_ms[1]) == 0  # point 1 stayed silent
    assert batch.v_a_mV[:, 1] == pytest.approx(solo1.v_a_mV, abs=1e-9)


def test_batch_carry_forward_matches_chained_calls():
    # Batched analogue of test_chunking.py's refractory-straddling regression:
    # point 0 has a spike cascade whose refractory period straddles the chunk split;
    # point 1 is silent throughout. Chaining two batched chunks (2nd seeded from the
    # 1st's returned per-point final state) must equal one unchunked batched call.
    duration_ms = 100.0
    split_ms = 40.0
    j_e_strong_mV = 500.0
    e_spikes_a = [[39.0], []]

    whole = simulate_pairs_batch(
        e_spikes_a=e_spikes_a, i_spikes_a=[[], []], e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )

    chunk_1_spikes = [[s for s in pt if s < split_ms] for pt in e_spikes_a]
    chunk_2_spikes = [[s - split_ms for s in pt if s >= split_ms] for pt in e_spikes_a]

    chunk_1 = simulate_pairs_batch(
        e_spikes_a=chunk_1_spikes, i_spikes_a=[[], []], e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=split_ms, dt_ms=DT_MS,
    )
    chunk_2 = simulate_pairs_batch(
        e_spikes_a=chunk_2_spikes, i_spikes_a=[[], []], e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms - split_ms, dt_ms=DT_MS,
        v_init_a_mV=chunk_1.v_a_final_mV, v_init_b_mV=chunk_1.v_b_final_mV,
        lastspike_init_a_ms=chunk_1.lastspike_a_final_ms - split_ms,
        lastspike_init_b_ms=chunk_1.lastspike_b_final_ms - split_ms,
        zi_e_a=chunk_1.zf_e_a, zi_i_a=chunk_1.zf_i_a,
        zi_e_b=chunk_1.zf_e_b, zi_i_b=chunk_1.zf_i_b,
    )

    chained_v_a0 = np.concatenate([chunk_1.v_a_mV[:, 0], chunk_2.v_a_mV[:, 0]])
    chained_spikes_a0 = np.concatenate([chunk_1.spikes_a_ms[0], chunk_2.spikes_a_ms[0] + split_ms])

    assert len(whole.spikes_a_ms[0]) >= 2
    assert whole.spikes_a_ms[0][0] < split_ms < whole.spikes_a_ms[0][0] + T_REF_MS
    assert chained_v_a0 == pytest.approx(whole.v_a_mV[:, 0], abs=1e-6)
    assert chained_spikes_a0 == pytest.approx(whole.spikes_a_ms[0], abs=1e-6)
    # Point 1 (silent) must also chain correctly (all-default carry-forward state).
    chained_v_a1 = np.concatenate([chunk_1.v_a_mV[:, 1], chunk_2.v_a_mV[:, 1]])
    assert chained_v_a1 == pytest.approx(whole.v_a_mV[:, 1], abs=1e-6)


def test_batch_chunked_matches_unchunked_batch_call():
    # Exercises simulate_pairs_batch_chunked directly -- the generator block1.run
    # actually drives -- rather than only the manually-chained carry-forward parameters
    # tested above. Point 0: the same refractory-straddling cascade; point 1: silent.
    duration_ms = 100.0
    chunk_duration_ms = 40.0  # doesn't divide 100 evenly -- exercises the ragged-last-chunk case too
    j_e_strong_mV = 500.0
    e_spikes_a = [[39.0], []]

    whole = simulate_pairs_batch(
        e_spikes_a=e_spikes_a, i_spikes_a=[[], []], e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=duration_ms, dt_ms=DT_MS,
    )

    def input_provider(start_ms, end_ms):
        e_a = [[s - start_ms for s in pt if start_ms <= s < end_ms] for pt in e_spikes_a]
        empty = [[] for _ in e_spikes_a]
        return e_a, empty, empty, empty

    v_a_chunks, spikes_a_chunks = [], [[] for _ in range(2)]
    for start_ms, end_ms, result in simulate_pairs_batch_chunked(
        n=2, input_provider=input_provider,
        j_e_mV=j_e_strong_mV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        total_duration_ms=duration_ms, chunk_duration_ms=chunk_duration_ms, dt_ms=DT_MS,
    ):
        v_a_chunks.append(result.v_a_mV)
        for k in range(2):
            spikes_a_chunks[k].append(result.spikes_a_ms[k] + start_ms)

    chained_v_a0 = np.concatenate([c[:, 0] for c in v_a_chunks])
    chained_spikes_a0 = np.concatenate(spikes_a_chunks[0])

    assert len(whole.spikes_a_ms[0]) >= 2
    assert chained_v_a0 == pytest.approx(whole.v_a_mV[:, 0], abs=1e-6)
    assert chained_spikes_a0 == pytest.approx(whole.spikes_a_ms[0], abs=1e-6)
    chained_v_a1 = np.concatenate([c[:, 1] for c in v_a_chunks])
    assert chained_v_a1 == pytest.approx(whole.v_a_mV[:, 1], abs=1e-6)


def test_default_carry_forward_state_matches_fresh_start():
    result = simulate_pairs_batch(
        e_spikes_a=[[10.0], [20.0]], i_spikes_a=[[], []],
        e_spikes_b=[[], []], i_spikes_b=[[], []],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV, tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS,
        theta_mV=THETA_MV, v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=DT_MS,
    )
    assert result.v_a_mV[0, 0] == pytest.approx(0.0)
    assert result.v_a_mV[0, 1] == pytest.approx(0.0)
