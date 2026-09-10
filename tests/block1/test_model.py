"""Seam: simulate_pair -- the two-neuron LIF+synapse postsynaptic-pair model (S-p.19)."""
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


def test_silence_produces_no_spikes_and_v_stays_at_rest():
    result = simulate_pair(
        e_spikes_a=[], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=0.01,
    )

    assert len(result.spikes_a_ms) == 0
    assert len(result.spikes_b_ms) == 0
    assert np.allclose(result.v_a_mV, 0.0)
    assert np.allclose(result.v_b_mV, 0.0)


def test_two_well_separated_e_spikes_onto_cell_a_each_peak_near_target_psp():
    # 90ms apart = 9*tau_m: the first PSP's residual has decayed to ~e^-9 (~1.2e-4) of its
    # peak by the time the second spike arrives, negligible against the 0.1% tolerance below
    # -- unlike a 30ms gap (only 3*tau_m), which left a ~5% residual that visibly biased the
    # second peak upward (caught while writing this test: 0.8245mV instead of 0.75mV, which
    # is *correct* linear superposition, not a bug -- 30ms just wasn't "well separated").
    result = simulate_pair(
        e_spikes_a=[10.0, 100.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=140.0, dt_ms=0.01,
    )

    t = result.t_ms
    v = result.v_a_mV
    peak_1 = v[(t >= 10.0) & (t < 25.0)].max()
    peak_2 = v[(t >= 100.0) & (t < 115.0)].max()

    assert peak_1 == pytest.approx(0.75, rel=5e-3)  # O(dt/tau_s) TimedArray ZOH bias, see model.py docstring
    assert peak_2 == pytest.approx(0.75, rel=5e-3)  # O(dt/tau_s) TimedArray ZOH bias, see model.py docstring


def test_inputs_onto_cell_a_do_not_affect_cell_b():
    # Heavy input onto cell A only -- cell B must stay completely silent. Catches an
    # accidental shared-state bug between the two cells' s_E/s_I/V variables.
    heavy_e_spikes_a = [float(t) for t in range(0, 200, 2)]  # 100 spikes, 2ms apart

    result = simulate_pair(
        e_spikes_a=heavy_e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=200.0, dt_ms=0.01,
    )

    assert result.v_a_mV.max() > 1.0  # sanity: cell A was actually driven up
    assert len(result.spikes_b_ms) == 0
    assert np.allclose(result.v_b_mV, 0.0)


def test_single_inhibitory_arrival_pushes_v_negative():
    # Confirms the "+J_E...-J_I..." sign convention (Phase 1) actually hyperpolarises
    # inside the pair model's synapse dynamics, not just the isolated calibration test.
    result = simulate_pair(
        e_spikes_a=[], i_spikes_a=[10.0], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=0.01,
    )

    assert result.v_a_mV.min() == pytest.approx(-0.75, rel=5e-3)  # O(dt/tau_s) TimedArray ZOH bias, see model.py docstring
    assert len(result.spikes_a_ms) == 0


def test_simultaneous_pooled_arrivals_in_the_same_timestep_both_register():
    # Pooled arrivals from independent presynaptic inputs routinely land in the same dt
    # bin (dt=0.01ms here; with hundreds of pooled Poisson inputs this is the common
    # case, not an edge case). Two E-arrivals at the identical timestamp must both count
    # -- the resulting peak should be ~2x the single-spike PSP peak (linear summation).
    result = simulate_pair(
        e_spikes_a=[10.0, 10.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=0.01,
    )

    assert result.v_a_mV.max() == pytest.approx(1.5, rel=5e-3)  # O(dt/tau_s) TimedArray ZOH bias, see model.py docstring


def test_inputs_onto_cell_b_do_not_affect_cell_a_and_produce_correct_peak():
    # Mirror of test_inputs_onto_cell_a_do_not_affect_cell_b -- the four synapse/generator
    # blocks in simulate_pair are structurally near-identical copy-paste differing only in
    # the target index string ("0" vs "1"); this is the seam that would catch a swapped
    # index (e.g. syn_e_b accidentally wired to j="0"), which nothing else exercises.
    result = simulate_pair(
        e_spikes_a=[], i_spikes_a=[], e_spikes_b=[10.0], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=0.01,
    )

    assert len(result.spikes_a_ms) == 0
    assert np.allclose(result.v_a_mV, 0.0)
    assert result.v_b_mV.max() == pytest.approx(0.75, rel=5e-3)  # O(dt/tau_s) TimedArray ZOH bias, see model.py docstring


def test_synaptic_current_trace_matches_j_e_times_s_e_right_after_a_spike():
    # I_syn(t) = J_E*s_E(t) - J_I*s_I(t) is the term fed to later phases' current
    # correlation `c` -- verify it directly, not just its downstream effect on V.
    # Immediately after one E spike, s_E jumps to 1, so I_syn should read J_E (~3.0mV),
    # decaying with tau_s=5ms thereafter; nothing in the suite checked this field before.
    result = simulate_pair(
        e_spikes_a=[10.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=J_E_MV, j_i_mV=J_I_MV,
        tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
        v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
        duration_ms=50.0, dt_ms=0.01,
    )

    t = result.t_ms
    i_syn = result.i_syn_a_mV
    just_after_spike = i_syn[(t >= 10.0) & (t < 10.05)]
    assert just_after_spike.max() == pytest.approx(J_E_MV, rel=1e-2)

    # 5ms later (one tau_s), should have decayed to ~1/e of that peak.
    one_tau_s_later = i_syn[(t >= 14.99) & (t < 15.01)][0]
    assert one_tau_s_later == pytest.approx(J_E_MV / np.e, rel=1e-2)


def test_spike_at_or_beyond_duration_is_rejected_not_silently_dropped():
    # Brian2 silently drops a SpikeGeneratorGroup event at/after the run length with no
    # error -- a real risk once Phase 3 generates arrival times independently of the
    # simulation window. Fail loudly instead of quietly undercounting synaptic drive.
    with pytest.raises(ValueError):
        simulate_pair(
            e_spikes_a=[49.5], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
            j_e_mV=J_E_MV, j_i_mV=J_I_MV,
            tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
            v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
            duration_ms=45.0, dt_ms=0.01,  # 49.5ms >= 45.0ms duration
        )
    with pytest.raises(ValueError):
        simulate_pair(
            e_spikes_a=[-1.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
            j_e_mV=J_E_MV, j_i_mV=J_I_MV,
            tau_m_ms=TAU_M_MS, tau_s_ms=TAU_S_MS, theta_mV=THETA_MV,
            v_reset_mV=V_RESET_MV, t_ref_ms=T_REF_MS,
            duration_ms=50.0, dt_ms=0.01,
        )
