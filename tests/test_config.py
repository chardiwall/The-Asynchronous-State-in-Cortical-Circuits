"""Shared config loader used by every block: no hardcoded constants."""
import pytest

from config import load_config


def test_load_config_reads_real_config_yaml():
    config = load_config("config.yaml")

    assert config["seed"] == 1234
    assert config["pair_model"]["neuron"]["tau_m_ms"] == 10.0
    assert config["pair_model"]["synapse"]["epsp_peak_mV"] == 0.75


def test_load_config_rejects_empty_file(tmp_path):
    # yaml.safe_load returns None for an empty file, breaking the "-> dict" contract;
    # a bare None causes a confusing TypeError far from the real cause (empty config.yaml
    # from a bad merge/partial write) wherever the caller does config["some_key"].
    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")

    with pytest.raises(ValueError):
        load_config(str(empty_file))
