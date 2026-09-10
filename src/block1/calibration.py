"""Synaptic-weight calibration for the Block 1 (Fig. 1) postsynaptic-pair model.

A single presynaptic spike drives ds/dt = -s/tau_s + delta(t), so s(t) = exp(-t/tau_s)
for t >= 0. The subthreshold membrane response to that one input, tau_m dV/dt = -V + J*s(t)
with V(0) = 0, is the standard double-exponential PSP (solved by convolving the s(t) kernel
with the membrane's own exponential impulse response -- verified independently via sympy
dsolve, a from-scratch Euler integration, and a Brian2 simulation; see session notes):

    V(t) = J * [tau_s/(tau_m-tau_s)] * (exp(-t/tau_m) - exp(-t/tau_s))
    t_peak = [tau_m*tau_s/(tau_m-tau_s)] * ln(tau_m/tau_s)

J is calibrated so V(t_peak) equals the target PSP peak (S-p.19: +0.75mV EPSP, 0.75mV IPSP
magnitude -- both J_E and J_I are positive; the membrane equation's own "+J_E...-J_I..."
structure supplies the inhibitory sign, see docs/paper/01-postsynaptic-pair.md).

simulate_psp_peak verifies this closed form against block1.model.simulate_pair (Phase 2's
actual pair-model dynamics), not a separate hand-built check.
"""
import math

from block1.model import simulate_pair


def psp_weight(tau_m_ms: float, tau_s_ms: float, target_peak_mV: float) -> float:
    if tau_s_ms >= tau_m_ms:
        raise ValueError(
            f"psp_weight requires tau_m_ms > tau_s_ms (double-exponential PSP is only "
            f"valid then); got tau_m_ms={tau_m_ms}, tau_s_ms={tau_s_ms}"
        )
    t_peak_ms = (tau_m_ms * tau_s_ms) / (tau_m_ms - tau_s_ms) * math.log(tau_m_ms / tau_s_ms)
    peak_per_unit_j = (tau_s_ms / (tau_m_ms - tau_s_ms)) * (
        math.exp(-t_peak_ms / tau_m_ms) - math.exp(-t_peak_ms / tau_s_ms)
    )
    return target_peak_mV / peak_per_unit_j


def simulate_psp_peak(
    j_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    dt_ms: float,
    duration_ms: float,
) -> float:
    """Drive cell A of the actual Phase 2 pair model (simulate_pair) with a single E
    spike at t=0, weight j_mV, and return the observed peak V in mV. Reuses simulate_pair
    directly (rather than a separate hand-built one-neuron model) so this check can never
    silently drift from the dynamics Phase 2 actually uses -- j_i_mV=j_mV is passed too
    but has no effect here since i_spikes_a is empty.
    """
    result = simulate_pair(
        e_spikes_a=[0.0],
        i_spikes_a=[],
        e_spikes_b=[],
        i_spikes_b=[],
        j_e_mV=j_mV,
        j_i_mV=j_mV,
        tau_m_ms=tau_m_ms,
        tau_s_ms=tau_s_ms,
        theta_mV=theta_mV,
        v_reset_mV=v_reset_mV,
        t_ref_ms=t_ref_ms,
        duration_ms=duration_ms,
        dt_ms=dt_ms,
    )
    return float(result.v_a_mV.max())


def calibrate_synaptic_weights(config: dict) -> tuple[float, float]:
    """J_E, J_I (both positive mV magnitudes) from config.yaml's pair_model section."""
    tau_m_ms = config["pair_model"]["neuron"]["tau_m_ms"]
    tau_s_ms = config["pair_model"]["synapse"]["tau_s_ms"]
    epsp_peak_mV = config["pair_model"]["synapse"]["epsp_peak_mV"]
    ipsp_peak_mV = config["pair_model"]["synapse"]["ipsp_peak_mV"]

    if epsp_peak_mV <= 0:
        raise ValueError(f"epsp_peak_mV must be positive, got {epsp_peak_mV}")
    if ipsp_peak_mV >= 0:
        raise ValueError(f"ipsp_peak_mV must be negative, got {ipsp_peak_mV}")

    j_e = psp_weight(tau_m_ms, tau_s_ms, epsp_peak_mV)
    j_i = psp_weight(tau_m_ms, tau_s_ms, -ipsp_peak_mV)
    return j_e, j_i
