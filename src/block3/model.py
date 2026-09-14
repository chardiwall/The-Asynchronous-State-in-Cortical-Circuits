"""Recurrent conductance-based spiking network (Fig. 3), docs/paper/03-recurrent-spiking-network.md.
Equations declared literally against SOM S-p.20-21 (ADR 0001): one neuron equation shared by
the E and I NeuronGroups (only N and t_ref differ), one difference-of-exponentials synapse
equation shared by all six ordered population pairs (only mean conductance, reversal potential
and delay range differ). X is a pure b2.PoissonGroup -- no membrane equation, no recurrent
input (docs/paper/03: "N_X=4000 independent Poisson trains").

X's own conduction delay isn't given a formula in the paper (only "from excitatory cells" and
"from inhibitory cells" ranges are stated); assumed to follow the excitatory-cell range since
V_rev^X=V_rev^E classifies X as excitatory-type (docs/paper/03 Ambiguity).

EE and II synapses exclude self-connections (i==j), matching block 2's binary-network
convention for within-population pairs -- a neuron does not synapse onto itself.
"""
from dataclasses import dataclass

import brian2 as b2
import numpy as np

from block3.connectivity import (
    generate_bernoulli_connectivity,
    resample_gaussian_conductances,
    sample_delays_ms,
)

NEURON_EQS = """
dV/dt = (-g_L*(V - V_L) + I_E + I_I + I_X + I_app) / C_m : volt (unless refractory)
I_E : amp
I_I : amp
I_X : amp
I_app : amp
"""


@dataclass
class Block3Network:
    network: b2.Network
    group_e: b2.NeuronGroup
    group_i: b2.NeuronGroup
    group_x: b2.PoissonGroup
    synapses: dict
    spikes_e: b2.SpikeMonitor
    spikes_i: b2.SpikeMonitor
    spikes_x: b2.SpikeMonitor


def _build_synapse(source, target, post_current_name, v_rev_mV, mean_g_nS, std_fraction,
                    delay_range_ms, tau_r, tau_d, tau_tilde, p, rng, name, exclude_self):
    V_rev = v_rev_mV * b2.mV
    eqs = f"""
    ds/dt = (x - s) / tau_d : 1 (clock-driven)
    dx/dt = -x / tau_r : 1 (clock-driven)
    g_syn : siemens
    {post_current_name}_post = -g_syn * s * (V_post - V_rev) : amp (summed)
    """
    # Explicit namespace, not Brian2's implicit frame-capture default: build_network()
    # constructs objects but a caller elsewhere calls .run() on the returned Network, in a
    # different frame -- implicit capture would look for tau_r/tau_d/... there and fail.
    namespace = {"tau_r": tau_r, "tau_d": tau_d, "tau_tilde": tau_tilde, "V_rev": V_rev}
    synapse = b2.Synapses(source, target, eqs, on_pre="x += tau_tilde / tau_r",
                           namespace=namespace, method="rk2", name=f"synapses_{name}")

    # Connectivity generated in numpy before connect() is called, not via Brian2's own
    # connect(p=...) + reading len(synapse) back afterwards -- required for cpp_standalone
    # (see PROGRESS.md Phase 2): standalone defers connect(p=...)'s resolution into generated
    # code, so Python can't learn the realised synapse count before that code has run.
    pre_index, post_index = generate_bernoulli_connectivity(len(source), len(target), p, rng, exclude_self)
    synapse.connect(i=pre_index, j=post_index)

    n = len(pre_index)
    synapse.g_syn = resample_gaussian_conductances(mean_g_nS, std_fraction, n, rng) * b2.nS
    low_ms, high_ms = delay_range_ms
    synapse.delay = sample_delays_ms(n, low_ms, high_ms, resolution_ms=0.05, rng=rng) * b2.ms
    return synapse


def build_network(params: dict, rng: np.random.Generator) -> Block3Network:
    b2.start_scope()
    # No b2.seed() needed: connectivity is now generated in numpy (generate_bernoulli_
    # connectivity), not Brian2's own connect(p=...), so `rng` alone controls every random
    # draw in this network (connectivity, conductance, delay) -- one RNG stream, not two.
    b2.defaultclock.dt = params["dt_ms"] * b2.ms

    n_e = params["populations"]["n_excitatory"]
    n_i = params["populations"]["n_inhibitory"]
    n_x = params["populations"]["n_external"]
    p = params["connection_probability"]

    neuron = params["neuron"]
    g_L = neuron["g_L_nS"] * b2.nS
    C_m = neuron["c_m_nF"] * b2.nF
    V_L = neuron["v_leak_mV"] * b2.mV
    theta = neuron["v_threshold_mV"] * b2.mV
    v_reset = neuron["v_reset_mV"] * b2.mV
    t_ref_e = neuron["t_refractory_excitatory_ms"] * b2.ms
    t_ref_i = neuron["t_refractory_inhibitory_ms"] * b2.ms

    # Explicit namespace (see _build_synapse's comment: construction and run() happen in
    # different frames, so Brian2's implicit frame-capture default would fail at run time).
    neuron_namespace = {"g_L": g_L, "C_m": C_m, "V_L": V_L, "theta": theta, "v_reset": v_reset}
    group_e = b2.NeuronGroup(n_e, NEURON_EQS, threshold="V>theta", reset="V=v_reset",
                              refractory=t_ref_e, namespace=neuron_namespace,
                              method="rk2", name="group_e")
    group_i = b2.NeuronGroup(n_i, NEURON_EQS, threshold="V>theta", reset="V=v_reset",
                              refractory=t_ref_i, namespace=neuron_namespace,
                              method="rk2", name="group_i")
    for group in (group_e, group_i):
        group.V = V_L
        group.I_app = 0 * b2.nA

    group_x = b2.PoissonGroup(
        n_x, rates=params["external_input"]["rate_hz"] * b2.Hz, name="group_x"
    )

    rev = params["reversal_potentials_mV"]
    cond = params["conductances_nS"]
    syn = params["synapse"]
    delays = params["delays_ms"]
    tau_r = syn["tau_rise_ms"] * b2.ms
    tau_d = syn["tau_decay_ms"] * b2.ms
    tau_tilde = syn["tau_normalisation_ms"] * b2.ms
    std_fraction = cond["heterogeneity_std_fraction"]

    # (source, target, postsynaptic current name, reversal potential, mean g, delay range, exclude self)
    specs = {
        "EE": (group_e, group_e, "I_E", rev["excitatory"], cond["g_EE"], delays["from_excitatory"], True),
        "IE": (group_e, group_i, "I_E", rev["excitatory"], cond["g_IE"], delays["from_excitatory"], False),
        "EI": (group_i, group_e, "I_I", rev["inhibitory"], cond["g_EI"], delays["from_inhibitory"], False),
        "II": (group_i, group_i, "I_I", rev["inhibitory"], cond["g_II"], delays["from_inhibitory"], True),
        "EX": (group_x, group_e, "I_X", rev["excitatory"], cond["g_EX"], delays["from_excitatory"], False),
        "IX": (group_x, group_i, "I_X", rev["excitatory"], cond["g_IX"], delays["from_excitatory"], False),
    }
    synapses = {
        name: _build_synapse(source, target, post_name, v_rev, mean_g, std_fraction,
                              delay_range, tau_r, tau_d, tau_tilde, p, rng, name, exclude_self)
        for name, (source, target, post_name, v_rev, mean_g, delay_range, exclude_self) in specs.items()
    }

    spikes_e = b2.SpikeMonitor(group_e, name="spikes_e")
    spikes_i = b2.SpikeMonitor(group_i, name="spikes_i")
    spikes_x = b2.SpikeMonitor(group_x, name="spikes_x")

    network = b2.Network(
        group_e, group_i, group_x, *synapses.values(), spikes_e, spikes_i, spikes_x
    )
    return Block3Network(
        network=network, group_e=group_e, group_i=group_i, group_x=group_x,
        synapses=synapses, spikes_e=spikes_e, spikes_i=spikes_i, spikes_x=spikes_x,
    )
