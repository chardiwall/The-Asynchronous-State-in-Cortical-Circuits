"""Seam: lsm.asynchrony.run_background -- one short real run, checked for the invariants the
report relies on, not for the paper's numbers (too short for those)."""
import numpy as np

from config import load_config
from lsm.asynchrony import run_background


def test_background_run_reports_rates_correlations_and_an_exact_current_decomposition():
    config = load_config("config.yaml")
    row = run_background(config, "faithful", n=250, network_index=0, length_s=1.0)
    assert row["rate_excitatory_hz"] > 0 and row["rate_inhibitory_hz"] > 0
    assert -1 < row["r_bar"] < 1 and row["sigma_r"] > 0
    parts = row["c_EE"] + row["c_II"] + row["c_XX"] + 2 * (row["c_EI"] + row["c_EX"] + row["c_IX"])
    assert np.isclose(parts, row["c_total"])
    assert row["c_EI"] < 0  # inhibitory and excitatory components anti-correlate: cancellation
