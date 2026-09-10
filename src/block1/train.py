"""Sweep runner for Block 1's parameter sweeps (Fig. 1B: p; Fig. 1E: r_in)."""
import numpy as np

from analysis import spike_count_correlation, stationary_correlation
from block1.chunked import simulate_pair_chunked
from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


def _run_point(
    p: float,
    r_in: float,
    n_e: int,
    n_i: int,
    rate_hz: float,
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
    chunk_duration_ms: float | None = None,
) -> dict:
    inputs = build_pair_inputs(
        n_e=n_e, n_i=n_i, p=p, r_in=r_in, rate_hz=rate_hz,
        duration_ms=duration_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
    )
    common_sim_kwargs = dict(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e_mV, j_i_mV=j_i_mV,
        tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms, theta_mV=theta_mV,
        v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms, dt_ms=dt_ms,
    )
    # chunk_duration_ms=None (default): simulate_pair directly, as every phase before
    # this used. Set it to run in bounded-memory chunks instead (ADR 0002's deferred
    # piece), needed at Fig. 1's real L=10,000s scale -- same result either way
    # (tests/block1/test_train.py::test_chunked_sweep_matches_unchunked_sweep).
    if chunk_duration_ms is None:
        result = simulate_pair(duration_ms=duration_ms, **common_sim_kwargs)
    else:
        result = simulate_pair_chunked(
            total_duration_ms=duration_ms, chunk_duration_ms=chunk_duration_ms,
            **common_sim_kwargs,
        )
    c = stationary_correlation(result.i_syn_a_mV, result.i_syn_b_mV)
    r_out = spike_count_correlation(
        result.spikes_a_ms, result.spikes_b_ms, duration_ms, bin_dt_ms, window_T_ms
    )
    return {"p": p, "r_in": r_in, "c": c, "r_out": r_out}


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
    chunk_duration_ms: float | None = None,
) -> list[dict]:
    """Fig. 1B: E-only, r_in fixed (0 for the paper's panel), p swept."""
    return [
        _run_point(
            p, r_in, n_e, 0, rate_hz, j_e_mV, j_i_mV, tau_m_ms, tau_s_ms, theta_mV,
            v_reset_mV, t_ref_ms, jitter_tau_ms, duration_ms, dt_ms, bin_dt_ms,
            window_T_ms, rng, chunk_duration_ms,
        )
        for p in p_values
    ]


def run_r_in_sweep(
    r_in_values: list[float],
    n_e: int,
    n_i: int,
    p: float,
    rate_hz: float,
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
    chunk_duration_ms: float | None = None,
) -> list[dict]:
    """Fig. 1E: p fixed (0.2), r_in swept. n_i=0 for the E-only curve, n_i=220 (with
    the E+I rate) for the E+I curve.
    """
    return [
        _run_point(
            p, r_in, n_e, n_i, rate_hz, j_e_mV, j_i_mV, tau_m_ms, tau_s_ms, theta_mV,
            v_reset_mV, t_ref_ms, jitter_tau_ms, duration_ms, dt_ms, bin_dt_ms,
            window_T_ms, rng, chunk_duration_ms,
        )
        for r_in in r_in_values
    ]
