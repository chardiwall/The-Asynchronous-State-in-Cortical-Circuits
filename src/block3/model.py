"""Recurrent conductance-based spiking network (Fig. 3), SOM S2.1.2, S-p.20-21.
Equations declared literally against SOM S-p.20-21: one neuron equation shared by
the E and I NeuronGroups (only N and t_ref differ), one difference-of-exponentials synapse
equation shared by all six ordered population pairs (only mean conductance, reversal potential
and delay range differ). X is a pure b2.PoissonGroup -- no membrane equation, no recurrent
input (SOM S-p.20-21: "N_X=4000 independent Poisson trains").

X's own conduction delay isn't given a formula in the paper (only "from excitatory cells" and
"from inhibitory cells" ranges are stated); assumed to follow the excitatory-cell range since
V_rev^X=V_rev^E classifies X as excitatory-type (an ambiguity in SOM S-p.20-21).

EE and II synapses exclude self-connections (i==j), matching block 2's binary-network
convention for within-population pairs -- a neuron does not synapse onto itself.

The synaptic kinetics live on the POSTSYNAPTIC neuron, one (s_c, x_c) pair per input channel
c in {E, I, X}, rather than on each of the ~9M synapses. This is exact, not an approximation:
the SOM's per-synapse equations are linear and tau_r, tau_d and V_rev depend only on the
presynaptic type, so S_c = sum_j g_ij s_ij obeys the same two ODEs driven by
tau_tilde * sum_j g_ij delta(t - t_j - d_ij). Each synapse keeps its own g_ij and delay d_ij;
on a spike it adds g_ij * tau_tilde/tau_r to x_c of its target. Integrating 5000 neurons
instead of 9M synapses per step is what makes the paper's run lengths reachable (~27 h per
200 s before). Side effect: the conductance term is now inside the neuron's rk2 step with V,
so the earlier caveat that Brian2's (summed) variables froze I_E/I_I/I_X across each step
(an effectively-Euler conductance term) no longer applies.
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
I_E = -s_E * (V - V_rev_E) : amp
I_I = -s_I * (V - V_rev_I) : amp
I_X = -s_X * (V - V_rev_E) : amp
ds_E/dt = (x_E - s_E) / tau_d : siemens
dx_E/dt = -x_E / tau_r : siemens
ds_I/dt = (x_I - s_I) / tau_d : siemens
dx_I/dt = -x_I / tau_r : siemens
ds_X/dt = (x_X - s_X) / tau_d : siemens
dx_X/dt = -x_X / tau_r : siemens
I_app : amp
theta : volt (constant)
"""


@dataclass
class Block3Network:
    network: b2.Network
    group_e: b2.NeuronGroup
    group_i: b2.NeuronGroup
    group_x: b2.Group  # PoissonGroup, or SpikeGeneratorGroup for explicit input events
    synapses: dict
    spikes_e: b2.SpikeMonitor
    spikes_i: b2.SpikeMonitor
    spikes_x: b2.SpikeMonitor


def _build_synapse(source, target, channel, mean_g_nS, std_fraction, delay_range_ms,
                   tau_r, tau_tilde, p, rng, name, exclude_self):
    # Explicit namespace, not Brian2's implicit frame-capture default: build_network()
    # constructs objects but a caller elsewhere calls .run() on the returned Network, in a
    # different frame -- implicit capture would look for tau_r/tau_tilde there and fail.
    synapse = b2.Synapses(source, target, "g_syn : siemens",
                          on_pre=f"x_{channel}_post += g_syn * tau_tilde / tau_r",
                          namespace={"tau_r": tau_r, "tau_tilde": tau_tilde},
                          name=f"synapses_{name}")

    # Connectivity generated in numpy before connect() is called, not via Brian2's own
    # connect(p=...) + reading len(synapse) back afterwards -- required for cpp_standalone
    # -- standalone defers connect(p=...)'s resolution into generated
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
    # theta is a per-neuron PARAMETER, not a namespace constant, so Fig. 3C-D can disable
    # the spiking mechanism in the recorded pair (S-p.21) by raising only those cells'
    # threshold out of reach. Every neuron is set to the config value below, so a normal
    # run is numerically identical to a single shared threshold.
    rev = params["reversal_potentials_mV"]
    cond = params["conductances_nS"]
    syn = params["synapse"]
    delays = params["delays_ms"]
    tau_r = syn["tau_rise_ms"] * b2.ms
    tau_d = syn["tau_decay_ms"] * b2.ms
    tau_tilde = syn["tau_normalisation_ms"] * b2.ms
    std_fraction = cond["heterogeneity_std_fraction"]

    neuron_namespace = {"g_L": g_L, "C_m": C_m, "V_L": V_L, "v_reset": v_reset,
                        "tau_r": tau_r, "tau_d": tau_d,
                        "V_rev_E": rev["excitatory"] * b2.mV, "V_rev_I": rev["inhibitory"] * b2.mV}
    group_e = b2.NeuronGroup(n_e, NEURON_EQS, threshold="V>theta", reset="V=v_reset",
                              refractory=t_ref_e, namespace=neuron_namespace,
                              method="rk2", name="group_e")
    group_i = b2.NeuronGroup(n_i, NEURON_EQS, threshold="V>theta", reset="V=v_reset",
                              refractory=t_ref_i, namespace=neuron_namespace,
                              method="rk2", name="group_i")
    for group in (group_e, group_i):
        group.V = V_L
        group.I_app = 0 * b2.nA
        group.theta = theta

    external = params["external_input"]
    if external["kind"] == "spike_generator":
        # Explicit input events (a test's single spike, or an event-camera stream), in
        # place of the SOM's background Poisson trains.
        group_x = b2.SpikeGeneratorGroup(n_x, external["indices"],
                                         np.asarray(external["times_ms"]) * b2.ms,
                                         name="group_x")
    else:
        group_x = b2.PoissonGroup(n_x, rates=external["rate_hz"] * b2.Hz, name="group_x")

    # (source, target, postsynaptic input channel, mean g, delay range, exclude self)
    specs = {
        "EE": (group_e, group_e, "E", cond["g_EE"], delays["from_excitatory"], True),
        "IE": (group_e, group_i, "E", cond["g_IE"], delays["from_excitatory"], False),
        "EI": (group_i, group_e, "I", cond["g_EI"], delays["from_inhibitory"], False),
        "II": (group_i, group_i, "I", cond["g_II"], delays["from_inhibitory"], True),
        "EX": (group_x, group_e, "X", cond["g_EX"], delays["from_excitatory"], False),
        "IX": (group_x, group_i, "X", cond["g_IX"], delays["from_excitatory"], False),
    }
    synapses = {
        name: _build_synapse(source, target, channel, mean_g, std_fraction, delay_range,
                             tau_r, tau_tilde, p, rng, name, exclude_self)
        for name, (source, target, channel, mean_g, delay_range, exclude_self) in specs.items()
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
