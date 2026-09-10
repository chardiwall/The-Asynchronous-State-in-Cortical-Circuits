"""Two-neuron postsynaptic-pair model (S-p.19), Block 1 / Fig. 1.

Per cell: tau_m dV/dt = -V + J_E*s_E(t) - J_I*s_I(t)  (if V < theta), spike/reset/refractory
at threshold. s_E = sum of individual E-synapse traces onto that cell; since every individual
trace obeys the same linear ds_i/dt = -s_i/tau_s + delta(t-t_i), their sum obeys the identical
ODE driven by the pooled (merged) set of all E-population arrival times onto that cell -- so
this model takes one pooled arrival-time array per cell per population, not per-input identity
(see session notes: this simplification was confirmed with the researcher before implementing).
"""
from dataclasses import dataclass

import brian2 as b2
import numpy as np


@dataclass
class PairResult:
    t_ms: np.ndarray
    v_a_mV: np.ndarray
    v_b_mV: np.ndarray
    i_syn_a_mV: np.ndarray
    i_syn_b_mV: np.ndarray
    spikes_a_ms: np.ndarray
    spikes_b_ms: np.ndarray


def _spike_generator(spike_times_ms: list[float]) -> b2.SpikeGeneratorGroup:
    # Pooled arrivals from many independent presynaptic inputs routinely land in the same
    # dt bin. Brian2 forbids one SpikeGeneratorGroup neuron firing twice in a timestep, so
    # give every pooled event its own virtual index (each fires at most once) rather than
    # reusing a single index 0 -- Synapses below connects them all to the same target cell.
    times = np.sort(np.asarray(spike_times_ms, dtype=float))
    indices = np.arange(len(times))
    return b2.SpikeGeneratorGroup(max(len(times), 1), indices, times * b2.ms)


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
                    f"Brian2 silently drops out-of-range SpikeGeneratorGroup events, so "
                    f"this is rejected explicitly rather than quietly undercounting input"
                )

    b2.start_scope()
    b2.defaultclock.dt = dt_ms * b2.ms

    tau_m = tau_m_ms * b2.ms
    tau_s = tau_s_ms * b2.ms
    J_E = j_e_mV * b2.mV
    J_I = j_i_mV * b2.mV
    theta = theta_mV * b2.mV
    v_reset = v_reset_mV * b2.mV
    t_ref = t_ref_ms * b2.ms

    eqs = """
    dV/dt = (-V + J_E*s_E - J_I*s_I)/tau_m : volt (unless refractory)
    ds_E/dt = -s_E/tau_s : 1
    ds_I/dt = -s_I/tau_s : 1
    """
    cells = b2.NeuronGroup(
        2, eqs, threshold="V>theta", reset="V=v_reset", refractory=t_ref, method="exact"
    )
    cells.V = 0 * b2.mV

    gen_e_a = _spike_generator(e_spikes_a)
    gen_i_a = _spike_generator(i_spikes_a)
    gen_e_b = _spike_generator(e_spikes_b)
    gen_i_b = _spike_generator(i_spikes_b)

    # Every pooled arrival is its own virtual source neuron (see _spike_generator); all of
    # them target the same single postsynaptic cell, so connect all-of-source to one j.
    # NOTE: j=0 (the integer) is falsy and trips Brian2's "must specify i, j or condition"
    # check -- use the string index-expression form for the connect target instead.
    syn_e_a = b2.Synapses(gen_e_a, cells, on_pre="s_E_post += 1")
    syn_e_a.connect(j="0")
    syn_i_a = b2.Synapses(gen_i_a, cells, on_pre="s_I_post += 1")
    syn_i_a.connect(j="0")
    syn_e_b = b2.Synapses(gen_e_b, cells, on_pre="s_E_post += 1")
    syn_e_b.connect(j="1")
    syn_i_b = b2.Synapses(gen_i_b, cells, on_pre="s_I_post += 1")
    syn_i_b.connect(j="1")

    state_mon = b2.StateMonitor(cells, ["V", "s_E", "s_I"], record=True)
    spike_mon = b2.SpikeMonitor(cells)

    b2.run(duration_ms * b2.ms)

    v = state_mon.V / b2.mV
    s_e = state_mon.s_E
    s_i = state_mon.s_I
    i_syn = j_e_mV * s_e - j_i_mV * s_i

    spike_times_ms = spike_mon.t / b2.ms
    spike_indices = spike_mon.i[:]

    return PairResult(
        t_ms=np.asarray(state_mon.t / b2.ms),
        v_a_mV=np.asarray(v[0]),
        v_b_mV=np.asarray(v[1]),
        i_syn_a_mV=np.asarray(i_syn[0]),
        i_syn_b_mV=np.asarray(i_syn[1]),
        spikes_a_ms=np.asarray(spike_times_ms[spike_indices == 0]),
        spikes_b_ms=np.asarray(spike_times_ms[spike_indices == 1]),
    )
