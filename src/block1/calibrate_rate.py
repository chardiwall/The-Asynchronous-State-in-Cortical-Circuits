"""E-only input-rate calibration (ambiguity 1, docs/paper/01-postsynaptic-pair.md): the
paper states 20Hz drives the E+I condition to 5Hz output but doesn't give the E-only
rate; it's found here by bisection, since there's no closed form for a LIF neuron's
output rate under Poisson bombardment. Assumes output rate increases monotonically with
input rate -- true for a purely excitatory drive.

Each output-rate measurement is itself a noisy Monte Carlo estimate (a single
simulate_pair run's spike count is Poisson-ish), which can violate bisection's
monotonicity assumption if the noise is comparable to tolerance_hz -- e.g. at L=5s,
target=5Hz, expected count ~25 has ~20% relative noise. n_trials averages independent
replicates at each candidate rate to reduce this by ~sqrt(n_trials) (a real bug caught
by review: the original single-sample version could converge to a systematically wrong
rate with no signal that anything had gone wrong).
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
    jitter_tau_ms: float,
    rate_low_hz: float,
    rate_high_hz: float,
    tolerance_hz: float,
    max_iterations: int,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    def single_trial_rate(trial_rate_hz: float) -> float:
        inputs = build_pair_inputs(
            n_e=n_e, n_i=0, p=0.0, r_in=0.0, rate_hz=trial_rate_hz,
            duration_ms=duration_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
        )
        result = simulate_pair(
            e_spikes_a=inputs.e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
            j_e_mV=j_e_mV, j_i_mV=j_i_mV,
            tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, theta_mV=theta_mV,
            v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=duration_ms, dt_ms=dt_ms,
        )
        return len(result.spikes_a_ms) / (duration_ms / 1000.0)

    def output_rate_at(trial_rate_hz: float) -> float:
        return float(np.mean([single_trial_rate(trial_rate_hz) for _ in range(n_trials)]))

    low, high = rate_low_hz, rate_high_hz
    rate_at_low = output_rate_at(low)
    rate_at_high = output_rate_at(high)
    if not (rate_at_low <= target_output_hz <= rate_at_high):
        raise ValueError(
            f"[rate_low_hz={low}, rate_high_hz={high}] does not bracket "
            f"target_output_hz={target_output_hz}: measured rates are "
            f"[{rate_at_low}, {rate_at_high}]. Widen the search bounds."
        )

    mid, rate = (low + high) / 2.0, None
    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        rate = output_rate_at(mid)
        if abs(rate - target_output_hz) < tolerance_hz:
            return mid
        if rate < target_output_hz:
            low = mid
        else:
            high = mid

    raise ValueError(
        f"calibrate_e_only_input_rate did not converge within {max_iterations} "
        f"iterations: last candidate {mid}Hz measured {rate}Hz against target "
        f"{target_output_hz}Hz (tolerance {tolerance_hz}Hz). Increase max_iterations, "
        f"n_trials, or duration_ms."
    )
