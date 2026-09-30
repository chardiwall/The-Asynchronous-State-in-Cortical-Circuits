"""Drives batched_model.simulate_pairs_batch repeatedly over a long run, carrying
V / lastspike / synaptic-trace-filter state forward across time chunks. Separated from
batched_model.py because the two answer different questions: that module simulates N
pairs for ONE window, this one turns that into an arbitrarily long run without ever
holding the whole trace in memory.
"""
import numpy as np

from block1.batched_model import simulate_pairs_batch


def chunk_boundaries(total_duration_ms: float, chunk_duration_ms: float):
    """(start_ms, end_ms) pairs tiling [0, total_duration_ms) in chunk_duration_ms-sized
    steps (final chunk shorter if it doesn't divide evenly).
    """
    start_ms = 0.0
    while start_ms < total_duration_ms:
        end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
        yield start_ms, end_ms
        start_ms = end_ms

def simulate_pairs_batch_chunked(
    n: int,
    input_provider,
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    total_duration_ms: float,
    chunk_duration_ms: float,
    dt_ms: float,
):
    """Yields (start_ms, end_ms, BatchPairResult) once per time chunk, carrying V/
    lastspike/synaptic-trace-filter state forward between chunks -- the batched,
    N-point analogue of the single-pair chunking this project's chunked.py used to do
    (deleted once its only caller, train.py's per-point sweep, was superseded by
    batching; see PROGRESS.md).

    input_provider(start_ms, end_ms) -> (e_spikes_a, i_spikes_a, e_spikes_b, i_spikes_b),
    each a length-n list of that chunk's pooled arrival times for point 0..n-1. Taking a
    callback rather than pre-built full-duration spike lists is what keeps memory bounded
    at production scale: block1.full_pass regenerates each point's inputs per chunk
    (Poisson processes have independent increments, so this is statistically exact, not
    an approximation -- the same reasoning the deleted train.py's _run_point_chunked
    established).

    This is a generator, not an accumulator: it does NOT concatenate results across
    chunks itself (holding e.g. the full L=10,000s current trace for all n points at
    once would defeat the point of chunking -- a real issue a review caught in an
    earlier version of this pipeline). The caller consumes each chunk's result as it's
    produced -- e.g. block1.full_pass folds it into a running analysis.StreamingCorrelation
    per point and only appends the much smaller spike-time arrays.
    """
    v_init_a, v_init_b = np.zeros(n), np.zeros(n)
    lastspike_init_a, lastspike_init_b = np.full(n, -1e4), np.full(n, -1e4)
    zi_e_a, zi_i_a = [None] * n, [None] * n
    zi_e_b, zi_i_b = [None] * n, [None] * n

    for start_ms, end_ms in chunk_boundaries(total_duration_ms, chunk_duration_ms):
        chunk_ms = end_ms - start_ms
        e_spikes_a, i_spikes_a, e_spikes_b, i_spikes_b = input_provider(start_ms, end_ms)

        result = simulate_pairs_batch(
            e_spikes_a=e_spikes_a, i_spikes_a=i_spikes_a,
            e_spikes_b=e_spikes_b, i_spikes_b=i_spikes_b,
            j_e_mV=j_e_mV, j_i_mV=j_i_mV, tau_m_ms=tau_m_ms, tau_s_ms=tau_s_ms,
            theta_mV=theta_mV, v_reset_mV=v_reset_mV, t_ref_ms=t_ref_ms,
            duration_ms=chunk_ms, dt_ms=dt_ms,
            v_init_a_mV=v_init_a, v_init_b_mV=v_init_b,
            lastspike_init_a_ms=lastspike_init_a, lastspike_init_b_ms=lastspike_init_b,
            zi_e_a=zi_e_a, zi_i_a=zi_i_a, zi_e_b=zi_e_b, zi_i_b=zi_i_b,
        )

        v_init_a, v_init_b = result.v_a_final_mV, result.v_b_final_mV
        # lastspike is local to each chunk's own t=0; re-base it to the next chunk's t=0
        # by subtracting this chunk's duration (see model.py's PairResult docstring).
        lastspike_init_a = result.lastspike_a_final_ms - chunk_ms
        lastspike_init_b = result.lastspike_b_final_ms - chunk_ms
        zi_e_a, zi_i_a, zi_e_b, zi_i_b = result.zf_e_a, result.zf_i_a, result.zf_e_b, result.zf_i_b

        yield start_ms, end_ms, result
