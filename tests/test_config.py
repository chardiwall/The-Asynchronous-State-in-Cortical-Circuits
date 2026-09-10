"""Shared config loader used by every block (CLAUDE.md: no hardcoded constants)."""
from config import load_config


def test_load_config_reads_real_config_yaml():
    config = load_config("config.yaml")

    assert config["seed"] == 1234
    assert config["pair_model"]["neuron"]["tau_m_ms"] == 10.0
    assert config["pair_model"]["synapse"]["epsp_peak_mV"] == 0.75
