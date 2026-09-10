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
"""
import math

import brian2 as b2


def psp_weight(tau_m_ms: float, tau_s_ms: float, target_peak_mV: float) -> float:
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
    dt_ms: float = 0.01,
    duration_ms: float = 50.0,
) -> float:
    """Drive one LIF neuron (S-p.19 equations) with a single spike at t=0, weight j_mV,
    and return the observed peak V in mV. Uses the same threshold/reset/refractory
    machinery Phase 2's pair model will use, even though a single 0.75mV-scale PSP never
    approaches threshold -- this is the seam that checks the closed form against the
    actual simulated dynamics, not just against itself.
    """
    b2.start_scope()
    b2.defaultclock.dt = dt_ms * b2.ms

    tau_m = tau_m_ms * b2.ms
    tau_s = tau_s_ms * b2.ms
    J = j_mV * b2.mV
    theta = theta_mV * b2.mV
    v_reset = v_reset_mV * b2.mV
    t_ref = t_ref_ms * b2.ms

    eqs = """
    dV/dt = (-V + J*s)/tau_m : volt (unless refractory)
    ds/dt = -s/tau_s : 1
    """
    neuron = b2.NeuronGroup(
        1, eqs, threshold="V>theta", reset="V=v_reset", refractory=t_ref, method="exact"
    )
    neuron.V = 0 * b2.mV

    spike_source = b2.SpikeGeneratorGroup(1, [0], [0 * b2.ms])
    synapse = b2.Synapses(spike_source, neuron, on_pre="s_post += 1")
    synapse.connect(i=0, j=0)

    monitor = b2.StateMonitor(neuron, "V", record=True)
    b2.run(duration_ms * b2.ms)

    return float(monitor.V[0].max() / b2.mV)


def calibrate_synaptic_weights(config: dict) -> tuple[float, float]:
    """J_E, J_I (both positive mV magnitudes) from config.yaml's pair_model section."""
    tau_m_ms = config["pair_model"]["neuron"]["tau_m_ms"]
    tau_s_ms = config["pair_model"]["synapse"]["tau_s_ms"]
    epsp_peak_mV = config["pair_model"]["synapse"]["epsp_peak_mV"]
    ipsp_peak_mV = config["pair_model"]["synapse"]["ipsp_peak_mV"]

    j_e = psp_weight(tau_m_ms, tau_s_ms, epsp_peak_mV)
    j_i = psp_weight(tau_m_ms, tau_s_ms, abs(ipsp_peak_mV))
    return j_e, j_i
