"""Runs simulate_pair over bounded time windows instead of one unchunked call spanning
the full duration (ADR 0002's deferred piece). At Fig. 1's real scale (L=10,000s,
dt=0.05ms -> 200M samples), one unchunked call needs ~15-25GB (four synaptic_trace
arrays, the TimedArray, Brian2's StateMonitor, ...); a bounded chunk_duration_ms keeps
peak memory proportional to chunk size instead of total duration. Correctness (must
equal one unchunked call) verified in tests/block1/test_chunked.py.
"""
import numpy as np

from block1.model import PairResult, simulate_pair


def _spikes_in_window(spikes: list[float], start_ms: float, end_ms: float) -> list[float]:
    return [s - start_ms for s in spikes if start_ms <= s < end_ms]


def simulate_pair_chunked(
    e_spikes_a: list[float],
    i_spikes_a: list[float],
    e_spikes_b: list[float],
    i_spikes_b: list[float],
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    total_duration_ms: float,
    dt_ms: float,
    chunk_duration_ms: float,
) -> PairResult:
    t_chunks, v_a_chunks, v_b_chunks = [], [], []
    i_syn_a_chunks, i_syn_b_chunks = [], []
    spikes_a_chunks, spikes_b_chunks = [], []

    v_init_a, v_init_b = 0.0, 0.0
    zi_e_a = zi_i_a = zi_e_b = zi_i_b = None

    start_ms = 0.0
    while start_ms < total_duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        result = simulate_pair(
            e_spikes_a=_spikes_in_window(e_spikes_a, start_ms, end_ms),
            i_spikes_a=_spikes_in_window(i_spikes_a, start_ms, end_ms),
            e_spikes_b=_spikes_in_window(e_spikes_b, start_ms, end_ms),
            i_spikes_b=_spikes_in_window(i_spikes_b, start_ms, end_ms),
            j_e_mV=j_e_mV, j_i_mV=j_i_mV, tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms,
            theta_mV=theta_mV, v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=end_ms - start_ms, dt_ms=dt_ms,
            v_init_a_mV=v_init_a, v_init_b_mV=v_init_b,
            zi_e_a=zi_e_a, zi_i_a=zi_i_a, zi_e_b=zi_e_b, zi_i_b=zi_i_b,
        )

        t_chunks.append(result.t_ms + start_ms)
        v_a_chunks.append(result.v_a_mV)
        v_b_chunks.append(result.v_b_mV)
        i_syn_a_chunks.append(result.i_syn_a_mV)
        i_syn_b_chunks.append(result.i_syn_b_mV)
        spikes_a_chunks.append(result.spikes_a_ms + start_ms)
        spikes_b_chunks.append(result.spikes_b_ms + start_ms)

        v_init_a, v_init_b = result.v_a_final_mV, result.v_b_final_mV
        zi_e_a, zi_i_a = result.zf_e_a, result.zf_i_a
        zi_e_b, zi_i_b = result.zf_e_b, result.zf_i_b
        start_ms = end_ms

    return PairResult(
        t_ms=np.concatenate(t_chunks),
        v_a_mV=np.concatenate(v_a_chunks),
        v_b_mV=np.concatenate(v_b_chunks),
        i_syn_a_mV=np.concatenate(i_syn_a_chunks),
        i_syn_b_mV=np.concatenate(i_syn_b_chunks),
        spikes_a_ms=np.concatenate(spikes_a_chunks),
        spikes_b_ms=np.concatenate(spikes_b_chunks),
        v_a_final_mV=v_init_a, v_b_final_mV=v_init_b,
        zf_e_a=zi_e_a, zf_i_a=zi_i_a, zf_e_b=zi_e_b, zf_i_b=zi_i_b,
    )
