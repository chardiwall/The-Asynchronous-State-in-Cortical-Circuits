"""Seam 1: closed-form synaptic-weight calibration.

Expected values are independently verified (not recomputed via the code under test) by
solving the ODE symbolically with sympy dsolve, cross-checked against a from-scratch
Euler integration and a Brian2 simulation (see session notes / commit history for the
derivation walkthrough):
V(t)/J = tau_s/(tau_m-tau_s) * (exp(-t/tau_m) - exp(-t/tau_s)),
t_peak = tau_m*tau_s/(tau_m-tau_s) * ln(tau_m/tau_s).
"""
import pytest

from block1.calibration import calibrate_synaptic_weights, psp_weight, simulate_psp_peak


def test_psp_weight_matches_paper_time_constants():
    # tau_m=10ms, tau_s=5ms (S-p.19): t_peak = 10*ln2 ~= 6.9315ms, V_peak/J = 0.25,
    # so a 0.75mV target implies J = 3.0mV. Confirmed by sympy dsolve, a plain-Euler
    # integration (dt=0.0001ms) and a Brian2 simulation, all agreeing to <0.01%.
    j = psp_weight(tau_m_ms=10.0, tau_s_ms=5.0, target_peak_mV=0.75)
    assert j == pytest.approx(3.0)


def test_psp_weight_generalises_to_other_time_constants():
    # tau_m=20ms, tau_s=5ms, target=1.0mV: t_peak ~= 9.24196ms, V_peak/J ~= 0.157490,
    # so J ~= 6.349604mV. Independently computed (fresh interpreter, not via psp_weight).
    j = psp_weight(tau_m_ms=20.0, tau_s_ms=5.0, target_peak_mV=1.0)
    assert j == pytest.approx(6.349604, rel=1e-5)


def test_simulated_psp_peak_matches_closed_form_calibration():
    # Independent check that the closed-form J from psp_weight, run through the actual
    # Brian2 LIF+synapse dynamics (S-p.19 equations, same as Phase 2 will use), produces
    # a peak PSP of the target 0.75mV -- not just correct arithmetic, but correct physics.
    tau_m_ms, tau_s_ms = 10.0, 5.0
    theta_mV, v_reset_mV, t_ref_ms = 20.0, 10.0, 2.0
    target_peak_mV = 0.75

    j = psp_weight(tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, target_peak_mV=target_peak_mV)
    peak_mV = simulate_psp_peak(
        j_mV=j,
        tau_m_ms=tau_m_ms,
        tau_s_ms=tau_s_ms,
        theta_mV=theta_mV,
        v_reset_mV=v_reset_mV,
        t_ref_ms=t_ref_ms,
    )

    assert peak_mV == pytest.approx(target_peak_mV, rel=1e-3)


def test_calibrate_synaptic_weights_reads_config_dict():
    # Minimal fixture mirroring config.yaml's pair_model section (not the real file --
    # this seam is unit-tested against a fixture; test_config.py separately checks the
    # real file's key paths match what this function reads).
    config = {
        "pair_model": {
            "neuron": {"tau_m_ms": 10.0},
            "synapse": {"tau_s_ms": 5.0, "epsp_peak_mV": 0.75, "ipsp_peak_mV": -0.75},
        }
    }

    j_e, j_i = calibrate_synaptic_weights(config)

    # Both positive magnitudes (session-confirmed sign convention): the membrane
    # equation's own "+J_E...-J_I..." structure supplies inhibition's sign.
    assert j_e == pytest.approx(3.0)
    assert j_i == pytest.approx(3.0)
