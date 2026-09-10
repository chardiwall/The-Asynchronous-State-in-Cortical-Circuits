"""Sweep runner for Block 1's parameter sweeps (Fig. 1B: p; Fig. 1E: r_in)."""
import numpy as np

from analysis import spike_count_correlation, stationary_correlation
from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


def run_p_sweep(
    p_values: list[float],
    n_e: int,
    rate_hz: float,
    r_in: float,
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    jitter_tau_ms: float,
    duration_ms: float,
    dt_ms: float,
    bin_dt_ms: float,
    window_T_ms: float,
    rng: np.random.Generator,
) -> list[dict]:
    """Fig. 1B: E-only, r_in=0 fixed, p swept. Returns one {p, c, r_out} dict per
    p_values entry, in order.
    """
    results = []
    for p in p_values:
        inputs = build_pair_inputs(
            n_e=n_e, n_i=0, p=p, r_in=r_in, rate_hz=rate_hz,
            duration_ms=duration_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
        )
        result = simulate_pair(
            e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
            e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
            j_e_mV=j_e_mV, j_i_mV=j_i_mV,
            tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, theta_mV=theta_mV,
            v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=duration_ms, dt_ms=dt_ms,
        )
        c = stationary_correlation(result.i_syn_a_mV, result.i_syn_b_mV)
        r_out = spike_count_correlation(
            result.spikes_a_ms, result.spikes_b_ms, duration_ms, bin_dt_ms, window_T_ms
        )
        results.append({"p": p, "c": c, "r_out": r_out})

    return results
