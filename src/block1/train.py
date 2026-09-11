"""Sweep runner for Block 1's parameter sweeps (Fig. 1B: p; Fig. 1E: r_in)."""
import numpy as np

from analysis import spike_count_correlation, stationary_correlation
from block1.dataset import build_pair_inputs
from block1.model import simulate_pair


def _run_point_chunked(
    p: float, r_in: float, n_e: int, n_i: int, rate_hz: float,
    j_e_mV: float, j_i_mV: float, tau_m_ms: float, tau_s_ms: float, theta_mV: float,
    v_reset_mV: float, t_ref_ms: float, jitter_tau_ms: float, duration_ms: float,
    dt_ms: float, rng: np.random.Generator, chunk_duration_ms: float,
):
    """Generates inputs AND runs Brian2/current-trace per chunk, not just the latter --
    at Fig. 1's real scale, E+I's pooled input arrays alone (~94M events -- more
    inputs *and* a higher rate than E-only's ~10M) are themselves too large to
    materialize for the full L=10,000s at once (confirmed empirically: a single-
    process full pass was OOM-killed entering the E+I sweep even with the Brian2 side
    already chunked). Regenerating inputs independently per chunk is statistically
    exact, not an approximation: a Poisson process has independent increments, so
    disjoint-interval realizations are distributionally identical to one continuous
    process (mother-train jitter's ~5-10ms range is negligible against a multi-hundred-
    second chunk, the same edge-effect argument already used elsewhere in this project).
    """
    v_init_a = v_init_b = 0.0
    zi_e_a = zi_i_a = zi_e_b = zi_i_b = None
    i_syn_a_chunks, i_syn_b_chunks, spikes_a_chunks, spikes_b_chunks = [], [], [], []

    start_ms = 0.0
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, duration_ms)
        chunk_ms = end_ms - start_ms
        inputs = build_pair_inputs(
            n_e=n_e, n_i=n_i, p=p, r_in=r_in, rate_hz=rate_hz,
            duration_ms=chunk_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
        )
        result = simulate_pair(
            e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
            e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
            j_e_mV=j_e_mV, j_i_mV=j_i_mV, tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms,
            theta_mV=theta_mV, v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=chunk_ms, dt_ms=dt_ms,
            v_init_a_mV=v_init_a, v_init_b_mV=v_init_b,
            zi_e_a=zi_e_a, zi_i_a=zi_i_a, zi_e_b=zi_e_b, zi_i_b=zi_i_b,
        )
        i_syn_a_chunks.append(result.i_syn_a_mV)
        i_syn_b_chunks.append(result.i_syn_b_mV)
        spikes_a_chunks.append(result.spikes_a_ms + start_ms)
        spikes_b_chunks.append(result.spikes_b_ms + start_ms)
        v_init_a, v_init_b = result.v_a_final_mV, result.v_b_final_mV
        zi_e_a, zi_i_a, zi_e_b, zi_i_b = result.zf_e_a, result.zf_i_a, result.zf_e_b, result.zf_i_b
        start_ms = end_ms

    return (
        np.concatenate(i_syn_a_chunks), np.concatenate(i_syn_b_chunks),
        np.concatenate(spikes_a_chunks), np.concatenate(spikes_b_chunks),
    )


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
    # chunk_duration_ms=None (default): generate inputs and simulate for the whole
    # duration in one shot, as every phase before this used. Set it to regenerate
    # inputs and simulate per bounded chunk instead (needed at Fig. 1's real
    # L=10,000s scale, see _run_point_chunked) -- statistically equivalent, not
    # bit-identical (tests/block1/test_train.py).
    if chunk_duration_ms is None:
        inputs = build_pair_inputs(
            n_e=n_e, n_i=n_i, p=p, r_in=r_in, rate_hz=rate_hz,
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
        i_syn_a_mV, i_syn_b_mV = result.i_syn_a_mV, result.i_syn_b_mV
        spikes_a_ms, spikes_b_ms = result.spikes_a_ms, result.spikes_b_ms
    else:
        i_syn_a_mV, i_syn_b_mV, spikes_a_ms, spikes_b_ms = _run_point_chunked(
            p, r_in, n_e, n_i, rate_hz, j_e_mV, j_i_mV, tau_m_ms, tau_s_ms, theta_mV,
            v_reset_mV, t_ref_ms, jitter_tau_ms, duration_ms, dt_ms, rng, chunk_duration_ms,
        )

    c = stationary_correlation(i_syn_a_mV, i_syn_b_mV)
    r_out = spike_count_correlation(spikes_a_ms, spikes_b_ms, duration_ms, bin_dt_ms, window_T_ms)
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
