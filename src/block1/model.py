"""Two-neuron postsynaptic-pair model (S-p.19), Block 1 / Fig. 1.

Per cell: tau_m dV/dt = -V + J_E*s_E(t) - J_I*s_I(t)  (if V < theta), spike/reset/refractory
at threshold. s_E = sum of individual E-synapse traces onto that cell; since every individual
trace obeys the same linear ds_i/dt = -s_i/tau_s + delta(t-t_i), their sum obeys the identical
ODE driven by the pooled (merged) set of all E-population arrival times onto that cell -- so
this model takes one pooled arrival-time array per cell per population, not per-input identity.

s_E(t)/s_I(t) are precomputed via block1.current_trace.synaptic_trace (ADR 0002) and fed to
Brian2 as TimedArrays, rather than built from per-event SpikeGeneratorGroup/Synapses objects
-- the latter doesn't scale to Fig. 1's real input volume (see ADR 0002). This is an internal
rework only; the public seam (pooled arrival arrays in, PairResult out) is unchanged.

Trade-off discovered while doing this rework (recorded in ADR 0002): TimedArray holds
s_E/s_I piecewise-constant between dt-grid samples, whereas the old mechanism let Brian2
exactly integrate V and the continuously-decaying s_E/s_I together within each timestep.
This introduces a small O(dt/tau_s) bias in V (~0.1% at dt=0.01ms, tau_s=5ms) that wasn't
present before -- V-derived quantities (PSP peaks, etc.) now need tolerances of a few
tenths of a percent rather than near-exact; i_syn_a_mV/i_syn_b_mV (taken directly from the
precomputed arrays, not from Brian2's integration) are unaffected and remain exact.
"""
from dataclasses import dataclass

import brian2 as b2
import numpy as np

from block1.current_trace import bin_edges, synaptic_trace


def _advance_one_step(v_last_mV: float, i_last_mV: float, dt_ms: float, tau_m_ms: float) -> float:
    """Brian2's StateMonitor records V *before* each step's update (when='start'), so
    v[-1] is the state at duration-dt, not duration -- missing exactly one step's
    evolution. For chunked runs (block1.chunked) this matters: the next chunk's initial
    V must be the state at the true chunk boundary. Closed-form exact update for
    tau_m*dV/dt=-V+I with I held constant over one dt (matching TimedArray's own
    zero-order hold): V(t+dt) = I + (V(t)-I)*exp(-dt/tau_m) -- the same formula Brian2's
    own method='exact' integrator uses for this linear ODE, verified to match Brian2's
    own output bit-for-bit in tests/block1/test_model_chunking.py.
    """
    a = np.exp(-dt_ms / tau_m_ms)
    return float(i_last_mV + (v_last_mV - i_last_mV) * a)


@dataclass
class PairResult:
    t_ms: np.ndarray
    v_a_mV: np.ndarray
    v_b_mV: np.ndarray
    i_syn_a_mV: np.ndarray
    i_syn_b_mV: np.ndarray
    spikes_a_ms: np.ndarray
    spikes_b_ms: np.ndarray
    # Carry-forward state for chunked runs (block1.chunked): the final V of each cell,
    # and the final synaptic_trace filter state (zf) of each of the 4 (cell, population)
    # traces. Pass these as the next chunk's v_init_*/zi_* to continue seamlessly.
    v_a_final_mV: float
    v_b_final_mV: float
    zf_e_a: np.ndarray
    zf_i_a: np.ndarray
    zf_e_b: np.ndarray
    zf_i_b: np.ndarray


def simulate_pair(
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
    duration_ms: float,
    dt_ms: float,
    v_init_a_mV: float = 0.0,
    v_init_b_mV: float = 0.0,
    zi_e_a: np.ndarray | None = None,
    zi_i_a: np.ndarray | None = None,
    zi_e_b: np.ndarray | None = None,
    zi_i_b: np.ndarray | None = None,
) -> PairResult:
    for name, spikes in (
        ("e_spikes_a", e_spikes_a),
        ("i_spikes_a", i_spikes_a),
        ("e_spikes_b", e_spikes_b),
        ("i_spikes_b", i_spikes_b),
    ):
        for spike_time in spikes:
            if not (0 <= spike_time < duration_ms):
                raise ValueError(
                    f"{name} contains {spike_time}ms, outside [0, {duration_ms}ms) -- "
                    f"rejected explicitly rather than quietly undercounting input"
                )

    bins = bin_edges(duration_ms, dt_ms)
    zi_by_name = {"e_a": zi_e_a, "i_a": zi_i_a, "e_b": zi_e_b, "i_b": zi_i_b}
    traces, zf_by_name = {}, {}
    for name, spikes in (
        ("e_a", e_spikes_a), ("i_a", i_spikes_a),
        ("e_b", e_spikes_b), ("i_b", i_spikes_b),
    ):
        traces[name], zf_by_name[name] = synaptic_trace(
            spikes, duration_ms, dt_ms, tau_s_ms, zi=zi_by_name[name], bins=bins
        )
    i_syn_a_mV = j_e_mV * traces["e_a"] - j_i_mV * traces["i_a"]
    i_syn_b_mV = j_e_mV * traces["e_b"] - j_i_mV * traces["i_b"]

    b2.start_scope()
    b2.defaultclock.dt = dt_ms * b2.ms

    tau_m = tau_m_ms * b2.ms
    theta = theta_mV * b2.mV
    v_reset = v_reset_mV * b2.mV
    t_ref = t_ref_ms * b2.ms

    # TimedArray's 2nd array dimension is indexed by neuron (`i`) when referenced as
    # I_drive(t, i) in the equation -- Brian2's native way to give each neuron in a group
    # its own time-varying input from one array, no per-event objects involved.
    i_values = np.stack([i_syn_a_mV, i_syn_b_mV], axis=1) * b2.mV
    i_drive = b2.TimedArray(i_values, dt=dt_ms * b2.ms)

    eqs = """
    dV/dt = (-V + i_drive(t, i))/tau_m : volt (unless refractory)
    """
    cells = b2.NeuronGroup(
        2, eqs, threshold="V>theta", reset="V=v_reset", refractory=t_ref, method="exact"
    )
    cells.V = [v_init_a_mV, v_init_b_mV] * b2.mV

    state_mon = b2.StateMonitor(cells, "V", record=True)
    spike_mon = b2.SpikeMonitor(cells)

    b2.run(duration_ms * b2.ms)

    v = state_mon.V / b2.mV
    spike_times_ms = spike_mon.t / b2.ms
    spike_indices = spike_mon.i[:]

    t_ms = np.asarray(state_mon.t / b2.ms)
    if len(t_ms) != len(i_syn_a_mV):
        raise ValueError(
            f"Brian2 produced {len(t_ms)} samples but synaptic_trace produced "
            f"{len(i_syn_a_mV)} for duration_ms={duration_ms}, dt_ms={dt_ms} -- their "
            f"sample-count conventions disagree for this (duration_ms, dt_ms) pair "
            f"(Brian2's Clock rounds differently near non-exact multiples). Use a "
            f"duration_ms that is an exact multiple of dt_ms."
        )

    return PairResult(
        t_ms=t_ms,
        v_a_mV=np.asarray(v[0]),
        v_b_mV=np.asarray(v[1]),
        i_syn_a_mV=i_syn_a_mV,
        i_syn_b_mV=i_syn_b_mV,
        spikes_a_ms=np.asarray(spike_times_ms[spike_indices == 0]),
        spikes_b_ms=np.asarray(spike_times_ms[spike_indices == 1]),
        v_a_final_mV=_advance_one_step(v[0][-1], i_syn_a_mV[-1], dt_ms, tau_m_ms),
        v_b_final_mV=_advance_one_step(v[1][-1], i_syn_b_mV[-1], dt_ms, tau_m_ms),
        zf_e_a=zf_by_name["e_a"],
        zf_i_a=zf_by_name["i_a"],
        zf_e_b=zf_by_name["e_b"],
        zf_i_b=zf_by_name["i_b"],
    )
