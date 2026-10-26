"""Seam: build_network -- the per-neuron conductance S_c = sum_j g_ij s_ij must equal the
superposition of the SOM's difference-of-exponentials kernel (S-p.20-21), one per incoming
spike, delayed by that synapse's own conduction delay. Truth is the closed form, not the code:
    s(u) = tau_tilde / (tau_d - tau_r) * (exp(-u/tau_d) - exp(-u/tau_r)),  u = t - t_spike - d
"""
import brian2 as b2
import numpy as np

from block3.model import build_network
from tests.block3.test_model import SMALL_PARAMS


def test_external_spike_produces_the_closed_form_conductance_in_every_target():
    spike_ms = 10.0
    params = {**SMALL_PARAMS,
              "external_input": {"kind": "spike_generator",
                                 "indices": [0], "times_ms": [spike_ms]}}
    net = build_network(params, rng=np.random.default_rng(7))
    monitor = b2.StateMonitor(net.group_e, "s_X", record=True, dt=0.05 * b2.ms)
    net.network.add(monitor)
    net.network.run(40 * b2.ms)

    t = np.asarray(monitor.t / b2.ms)
    syn = net.synapses["EX"]
    pre, post = np.asarray(syn.i), np.asarray(syn.j)
    g_nS, d_ms = np.asarray(syn.g_syn / b2.nS), np.asarray(syn.delay / b2.ms)
    tau_r, tau_d, tau_tilde = 1.0, 5.0, 1.0

    expected = np.zeros((SMALL_PARAMS["populations"]["n_excitatory"], len(t)))
    for k in np.nonzero(pre == 0)[0]:
        # Brian2 delivers in the `synapses` slot, after the step's state update and its
        # `start`-slot monitor, so the recorded trace lags the continuous kernel by one dt.
        u = t - spike_ms - d_ms[k] - 0.05
        kernel = tau_tilde / (tau_d - tau_r) * (np.exp(-u / tau_d) - np.exp(-u / tau_r))
        expected[post[k]] += g_nS[k] * np.where(u > 0, kernel, 0.0)

    measured = np.asarray(monitor.s_X / b2.nS)
    assert expected.max() > 0.5            # the spike reached somebody
    assert np.max(np.abs(measured - expected)) < 0.01 * expected.max()  # rk2 error only
    assert abs(measured.sum() - expected.sum()) < 0.02 * expected.sum()
