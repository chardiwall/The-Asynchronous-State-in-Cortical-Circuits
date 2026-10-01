"""Seam: build_network -- wires the E/I/X populations and six synapse types together per
SOM S2.1.2, S-p.20-21. Small N throughout (test speed); behavior only,
not Brian2's own equation-integration correctness.
"""
import brian2 as b2
import numpy as np

from block3.model import build_network

SMALL_PARAMS = {
    "populations": {"n_excitatory": 40, "n_inhibitory": 20, "n_external": 40},
    "connection_probability": 0.2,
    "neuron": {
        "c_m_nF": 0.25, "g_L_nS": 16.7, "v_leak_mV": -70.0, "v_threshold_mV": -50.0,
        "v_reset_mV": -60.0, "t_refractory_excitatory_ms": 2.0, "t_refractory_inhibitory_ms": 1.0,
    },
    "reversal_potentials_mV": {"excitatory": 0.0, "inhibitory": -80.0},
    "conductances_nS": {
        "g_EE": 2.4, "g_EI": 40.0, "g_IE": 4.8, "g_II": 40.0, "g_EX": 5.4, "g_IX": 5.4,
        "heterogeneity_std_fraction": 0.5,
    },
    "synapse": {"tau_rise_ms": 1.0, "tau_decay_ms": 5.0, "tau_normalisation_ms": 1.0},
    "delays_ms": {"from_excitatory": [0.5, 1.5], "from_inhibitory": [0.1, 0.9]},
    "external_input": {"kind": "poisson", "rate_hz": 2.5},
    "dt_ms": 0.05,
}


def test_synapse_counts_land_near_p_times_pre_times_post_and_conductances_are_nonnegative():
    net = build_network(SMALL_PARAMS, rng=np.random.default_rng(4))

    n_e, n_i, n_x = 40, 20, 40
    p = 0.2
    expected_counts = {
        "EE": n_e * n_e * p, "EI": n_e * n_i * p, "IE": n_i * n_e * p, "II": n_i * n_i * p,
        "EX": n_e * n_x * p, "IX": n_i * n_x * p,
    }
    for name, expected in expected_counts.items():
        synapse = net.synapses[name]
        assert abs(len(synapse) - expected) < 4 * np.sqrt(expected)  # Bernoulli(p) count
        assert np.all(np.asarray(synapse.g_syn) >= 0.0)


def test_running_produces_no_nan_and_every_population_spikes_under_strong_drive():
    # Test-only: boosted external rate, not the paper's 2.5 Hz -- at this small N the
    # network isn't in the paper's target operating point anyway (N can't be shrunk
    # without breaking it), so this only checks that the wiring *can* deliver
    # enough drive to spike, not that rates are realistic.
    params = {**SMALL_PARAMS, "external_input": {"kind": "poisson", "rate_hz": 200.0}}
    net = build_network(params, rng=np.random.default_rng(5))

    net.network.run(500 * b2.ms)

    assert not np.any(np.isnan(np.asarray(net.group_e.V / b2.mV)))
    assert not np.any(np.isnan(np.asarray(net.group_i.V / b2.mV)))
    assert net.spikes_e.num_spikes > 0
    assert net.spikes_i.num_spikes > 0
    assert net.spikes_x.num_spikes > 0
