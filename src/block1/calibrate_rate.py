"""E-only input-rate calibration (ambiguity 1, docs/paper/01-postsynaptic-pair.md): the
paper states 20Hz drives the E+I condition to 5Hz output but doesn't give the E-only
rate; it's found here by bisection, since there's no closed form for a LIF neuron's
output rate under Poisson bombardment. Assumes output rate increases monotonically with
input rate -- true for a purely excitatory drive.
"""
import numpy as np

from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


def calibrate_e_only_input_rate(
    target_output_hz: float,
    n_e: int,
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    duration_ms: float,
    dt_ms: float,
    rate_low_hz: float,
    rate_high_hz: float,
    tolerance_hz: float,
    max_iterations: int,
    rng: np.random.Generator,
) -> float:
    def output_rate_at(trial_rate_hz: float) -> float:
        inputs = build_pair_inputs(
            n_e=n_e, n_i=0, p=0.0, r_in=0.0, rate_hz=trial_rate_hz,
            duration_ms=duration_ms, jitter_tau_ms=5.0, rng=rng,
        )
        result = simulate_pair(
            e_spikes_a=inputs.e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
            j_e_mV=j_e_mV, j_i_mV=j_i_mV,
            tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, theta_mV=theta_mV,
            v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=duration_ms, dt_ms=dt_ms,
        )
        return len(result.spikes_a_ms) / (duration_ms / 1000.0)

    low, high = rate_low_hz, rate_high_hz
    mid = (low + high) / 2.0
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        rate = output_rate_at(mid)
        if abs(rate - target_output_hz) < tolerance_hz:
            return mid
        if rate < target_output_hz:
            low = mid
        else:
            high = mid

    return mid
