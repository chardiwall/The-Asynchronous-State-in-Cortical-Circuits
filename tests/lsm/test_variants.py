"""Seam: lsm.variants.network_params / column_connect -- how each LSM variant is derived
from block 3's paper network. Truth: SOM S-p.20-21's numbers and the balanced-state
sqrt(K) scaling rule; Maass, Natschlaeger & Markram (2002)'s connection rule."""
import numpy as np
import pytest

from config import load_config
from lsm.variants import column_connect, network_params

CONFIG = load_config("config.yaml")


def test_faithful_variant_at_the_paper_size_is_block_3_itself():
    params = network_params(CONFIG, "faithful", n=5000, n_external=4000)
    assert params["populations"] == {"n_excitatory": 4000, "n_inhibitory": 1000, "n_external": 4000}
    assert params["conductances_nS"]["g_EE"] == pytest.approx(2.4)
    assert params["conductances_nS"]["g_EI"] == pytest.approx(40.0)
    assert params["conductances_nS"]["g_IX"] == pytest.approx(5.4)
    assert params["connection_probability"] == 0.2


def test_faithful_variant_scales_each_conductance_by_sqrt_of_its_presynaptic_shrinkage():
    params = network_params(CONFIG, "faithful", n=500, n_external=1000)
    g = params["conductances_nS"]
    assert params["populations"] == {"n_excitatory": 400, "n_inhibitory": 100, "n_external": 1000}
    assert g["g_EE"] == pytest.approx(2.4 * np.sqrt(10))     # 4000 E -> 400 E
    assert g["g_II"] == pytest.approx(40.0 * np.sqrt(10))    # 1000 I -> 100 I
    assert g["g_EX"] == pytest.approx(5.4 * np.sqrt(4))      # 4000 X -> 1000 X


def test_weak_variants_keep_paper_recurrent_conductances_but_share_the_scaled_input_path():
    # Every variant gets the same sqrt-scaled X -> reservoir pathway times its calibrated
    # gain, so variants differ only in their recurrent wiring and coupling.
    for name in ("dense_weak", "maass"):
        params = network_params(CONFIG, name, n=500, n_external=400)
        gain = CONFIG["lsm"]["input_gain"][name]
        assert params["conductances_nS"]["g_EE"] == pytest.approx(2.4)
        assert params["conductances_nS"]["g_II"] == pytest.approx(40.0)
        assert params["conductances_nS"]["g_EX"] == pytest.approx(5.4 * np.sqrt(10) * gain)


def test_column_connection_probability_follows_maass_gaussian_of_distance():
    # 4 sites in a row, one E neuron each; C_EE * exp(-(D/lambda)^2) at D = 1, 2, 3.
    lam, c_ee = CONFIG["lsm"]["column"]["lambda"], CONFIG["lsm"]["column"]["c"]["EE"]
    positions = {"E": np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]), "I": np.zeros((0, 3))}
    connect = column_connect(CONFIG, positions, p_external=0.2)
    counts = np.zeros((4, 4))
    rng = np.random.default_rng(0)
    for _ in range(4000):
        pre, post = connect("EE", 4, 4, rng, exclude_self=True)
        np.add.at(counts, (pre, post), 1)
    for d in (1, 2, 3):
        assert counts[0, d] / 4000 == pytest.approx(c_ee * np.exp(-(d / lam) ** 2), abs=0.02)
    assert counts[0, 0] == 0  # no self-connections
