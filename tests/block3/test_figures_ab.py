"""Seam: run_one_task -- one independent Slurm array task per network realisation
(PROGRESS.md Phase 2), mirroring block2's full_pass.py (one process per task, no
in-process pool). Small N / short duration for test speed."""
import numpy as np

from block3.figures_ab import run_one_task

SMALL_CONFIG = {
    "seed": 42,
    "analysis": {"bin_dt_ms": 1.0, "count_window_T_ms": 50.0},
    "spiking_network": {
        "populations": {"n_excitatory": 40, "n_inhibitory": 20, "n_external": 40},
        "connection_probability": 0.2,
        "neuron": {
            "c_m_nF": 0.25, "g_L_nS": 16.7, "v_leak_mV": -70.0, "v_threshold_mV": -50.0,
            "v_reset_mV": -60.0, "t_refractory_excitatory_ms": 2.0,
            "t_refractory_inhibitory_ms": 1.0,
        },
        "reversal_potentials_mV": {"excitatory": 0.0, "inhibitory": -80.0},
        "conductances_nS": {
            "g_EE": 2.4, "g_EI": 40.0, "g_IE": 4.8, "g_II": 40.0, "g_EX": 5.4, "g_IX": 5.4,
            "heterogeneity_std_fraction": 0.5,
        },
        "synapse": {"tau_rise_ms": 1.0, "tau_decay_ms": 5.0, "tau_normalisation_ms": 1.0},
        "delays_ms": {"from_excitatory": [0.5, 1.5], "from_inhibitory": [0.1, 0.9]},
        "external_input": {"kind": "poisson", "rate_hz": 200.0},  # boosted, test speed only
        "burn_in_ms": 50.0,
        "simulation": {"dt_ms": 0.05},
        "r_bar_sample_size": 10,  # tiny, real full-pass config uses 1000
    },
}


def test_returns_rates_and_r_bar_with_no_nan():
    result = run_one_task(network_index=0, config=SMALL_CONFIG, duration_ms=200.0)

    assert result["network_index"] == 0
    assert result["rate_excitatory_hz"] > 0
    assert result["rate_inhibitory_hz"] > 0
    assert np.isfinite(result["r_bar_EE"])
    assert result["nan_free"] is True


def test_different_network_index_uses_a_different_seed_and_gives_different_result():
    result_0 = run_one_task(network_index=0, config=SMALL_CONFIG, duration_ms=200.0)
    result_1 = run_one_task(network_index=1, config=SMALL_CONFIG, duration_ms=200.0)

    assert result_0["rate_excitatory_hz"] != result_1["rate_excitatory_hz"]
