"""Batches N independent postsynaptic pairs into ONE Brian2 simulation (a NeuronGroup of
size 2N) instead of N separate model.simulate_pair calls.

Why this is exact, not an approximation: the N points are physically independent --
each is its own feedforward pair with its own pooled inputs, no coupling between points
(model.py's docstring: "no recurrent connectivity... purely feedforward"). Stacking them
into one NeuronGroup changes nothing about any individual point's dynamics; it only
changes how many neurons share one Brian2 Network/Clock. This targets the documented
bottleneck directly: a benchmark found input generation is fast
(~2s for 9.4M events) but Brian2's own per-run cost dominates (~260s/1000s of dynamics)
-- N separate b2.run() calls each pay that per-run overhead once; one batched call pays
it once total, and (unlike the unbatched path) hands the GPU backend (brian2cuda) 2N-wide
per-timestep work instead of 2-wide.

Same equations, same TimedArray zero-order-hold, same threshold/reset/
refractory as model.simulate_pair -- this module changes N, not the math. The
carry-forward state model.py added for chunking (V, lastspike, synaptic-trace filter
state) generalizes here from scalars/1-D arrays to length-N arrays/lists, one entry per
point, using the same "read Brian2's own live NeuronGroup state after b2.run()" approach
model.py's PairResult docstring explains.
"""
from dataclasses import dataclass

import brian2 as b2
import numpy as np

from lib.psc import bin_edges, synaptic_trace



@dataclass
class BatchPairResult:
    t_ms: np.ndarray            # (T,)
    v_a_mV: np.ndarray          # (T, N)
    v_b_mV: np.ndarray          # (T, N)
    i_syn_a_mV: np.ndarray      # (T, N)
    i_syn_b_mV: np.ndarray      # (T, N)
    spikes_a_ms: list           # length N, each an np.ndarray of that point's cell-A spike times
    spikes_b_ms: list           # length N
    v_a_final_mV: np.ndarray    # (N,) -- carry-forward state, see module docstring
    v_b_final_mV: np.ndarray    # (N,)
    lastspike_a_final_ms: np.ndarray  # (N,)
    lastspike_b_final_ms: np.ndarray  # (N,)
    zf_e_a: list                # length N, each that point's synaptic_trace filter state
    zf_i_a: list
    zf_e_b: list
    zf_i_b: list


def simulate_pairs_batch(
    e_spikes_a: list[list[float]],
    i_spikes_a: list[list[float]],
    e_spikes_b: list[list[float]],
    i_spikes_b: list[list[float]],
    j_e_mV: float,
    j_i_mV: float,
    tau_m_ms: float,
    tau_s_ms: float,
    theta_mV: float,
    v_reset_mV: float,
    t_ref_ms: float,
    duration_ms: float,
    dt_ms: float,
    v_init_a_mV: np.ndarray | None = None,
    v_init_b_mV: np.ndarray | None = None,
    lastspike_init_a_ms: np.ndarray | None = None,
    lastspike_init_b_ms: np.ndarray | None = None,
    zi_e_a: list | None = None,
    zi_i_a: list | None = None,
    zi_e_b: list | None = None,
    zi_i_b: list | None = None,
) -> BatchPairResult:
    n = len(e_spikes_a)
    for name, spikes_by_point in (
        ("i_spikes_a", i_spikes_a), ("e_spikes_b", e_spikes_b), ("i_spikes_b", i_spikes_b),
    ):
        if len(spikes_by_point) != n:
            raise ValueError(f"{name} has {len(spikes_by_point)} points, expected {n}")

    v_init_a_mV = np.zeros(n) if v_init_a_mV is None else np.asarray(v_init_a_mV)
    v_init_b_mV = np.zeros(n) if v_init_b_mV is None else np.asarray(v_init_b_mV)
    lastspike_init_a_ms = np.full(n, -1e4) if lastspike_init_a_ms is None else np.asarray(lastspike_init_a_ms)
    lastspike_init_b_ms = np.full(n, -1e4) if lastspike_init_b_ms is None else np.asarray(lastspike_init_b_ms)
    zi_e_a = zi_e_a or [None] * n
    zi_i_a = zi_i_a or [None] * n
    zi_e_b = zi_e_b or [None] * n
    zi_i_b = zi_i_b or [None] * n

    bins = bin_edges(duration_ms, dt_ms)
    n_samples = len(bins) - 1
    i_syn_a_mV = np.empty((n_samples, n))
    i_syn_b_mV = np.empty((n_samples, n))
    zf_e_a, zf_i_a, zf_e_b, zf_i_b = [None] * n, [None] * n, [None] * n, [None] * n

    for k in range(n):
        for name, spikes in (
            ("e_spikes_a", e_spikes_a[k]), ("i_spikes_a", i_spikes_a[k]),
            ("e_spikes_b", e_spikes_b[k]), ("i_spikes_b", i_spikes_b[k]),
        ):
            arr = np.asarray(spikes, dtype=float)
            if arr.size and not ((arr >= 0).all() and (arr < duration_ms).all()):
                bad = arr[(arr < 0) | (arr >= duration_ms)][0]
                raise ValueError(
                    f"point {k}'s {name} contains {bad}ms, outside "
                    f"[0, {duration_ms}ms) -- rejected explicitly rather than "
                    f"quietly undercounting input"
                )
        s_e_a, zf_e_a[k] = synaptic_trace(e_spikes_a[k], duration_ms, dt_ms, tau_s_ms, zi=zi_e_a[k], bins=bins)
        s_i_a, zf_i_a[k] = synaptic_trace(i_spikes_a[k], duration_ms, dt_ms, tau_s_ms, zi=zi_i_a[k], bins=bins)
        s_e_b, zf_e_b[k] = synaptic_trace(e_spikes_b[k], duration_ms, dt_ms, tau_s_ms, zi=zi_e_b[k], bins=bins)
        s_i_b, zf_i_b[k] = synaptic_trace(i_spikes_b[k], duration_ms, dt_ms, tau_s_ms, zi=zi_i_b[k], bins=bins)
        i_syn_a_mV[:, k] = j_e_mV * s_e_a - j_i_mV * s_i_a
        i_syn_b_mV[:, k] = j_e_mV * s_e_b - j_i_mV * s_i_b

    b2.start_scope()
    b2.defaultclock.dt = dt_ms * b2.ms

    tau_m = tau_m_ms * b2.ms
    theta = theta_mV * b2.mV
    v_reset = v_reset_mV * b2.mV
    t_ref = t_ref_ms * b2.ms

    # Neuron layout: 0..n-1 = cell A of points 0..n-1; n..2n-1 = cell B of points 0..n-1.
    i_values = np.concatenate([i_syn_a_mV, i_syn_b_mV], axis=1) * b2.mV
    i_drive = b2.TimedArray(i_values, dt=dt_ms * b2.ms)

    eqs = """
    dV/dt = (-V + i_drive(t, i))/tau_m : volt (unless refractory)
    """
    cells = b2.NeuronGroup(
        2 * n, eqs, threshold="V>theta", reset="V=v_reset", refractory=t_ref, method="exact"
    )
    v_init_mV = np.concatenate([v_init_a_mV, v_init_b_mV])
    lastspike_init_ms = np.concatenate([lastspike_init_a_ms, lastspike_init_b_ms])
    cells.V = v_init_mV * b2.mV
    # lastspike alone carries refractory state across a chunk boundary; see model.py for
    # why `not_refractory` is not set beside it (Brian2 recomputes it every step).
    cells.lastspike = lastspike_init_ms * b2.ms

    state_mon = b2.StateMonitor(cells, "V", record=True)
    spike_mon = b2.SpikeMonitor(cells)

    b2.run(duration_ms * b2.ms)

    v = state_mon.V / b2.mV
    spike_times_ms = np.asarray(spike_mon.t / b2.ms)
    spike_indices = np.asarray(spike_mon.i[:])

    t_ms = np.asarray(state_mon.t / b2.ms)
    if len(t_ms) != n_samples:
        raise ValueError(
            f"Brian2 produced {len(t_ms)} samples but synaptic_trace produced "
            f"{n_samples} for duration_ms={duration_ms}, dt_ms={dt_ms} -- their "
            f"sample-count conventions disagree for this (duration_ms, dt_ms) pair. "
            f"Use a duration_ms that is an exact multiple of dt_ms."
        )

    spikes_a_ms = [spike_times_ms[spike_indices == k] for k in range(n)]
    spikes_b_ms = [spike_times_ms[spike_indices == n + k] for k in range(n)]

    v_final_mV = np.asarray(cells.V / b2.mV)
    lastspike_final_ms = np.asarray(cells.lastspike / b2.ms)

    return BatchPairResult(
        t_ms=t_ms,
        v_a_mV=np.asarray(v[:n]).T,
        v_b_mV=np.asarray(v[n:]).T,
        i_syn_a_mV=i_syn_a_mV,
        i_syn_b_mV=i_syn_b_mV,
        spikes_a_ms=spikes_a_ms,
        spikes_b_ms=spikes_b_ms,
        v_a_final_mV=v_final_mV[:n],
        v_b_final_mV=v_final_mV[n:],
        lastspike_a_final_ms=lastspike_final_ms[:n],
        lastspike_b_final_ms=lastspike_final_ms[n:],
        zf_e_a=zf_e_a, zf_i_a=zf_i_a, zf_e_b=zf_e_b, zf_i_b=zf_i_b,
    )
