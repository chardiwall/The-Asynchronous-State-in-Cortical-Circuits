"""End-to-end checks against the real config.yaml (not a fixture) -- this is the
committed, reproducible source for the "ran end-to-end, NaN-checked" claims in
PROGRESS.md. Also appends to artifacts/ so there's a log trail of real runs.
"""
import datetime
import json

import numpy as np
import pytest

from block1.calibration import calibrate_synaptic_weights
from block1.model import simulate_pair
from config import load_config


def _log(record: dict) -> None:
    with open("artifacts/metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    with open("artifacts/run.log", "a") as f:
        f.write(f"{record['timestamp']} {record['phase']}: {record}\n")


def test_phase1_calibration_against_real_config():
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]

    j_e, j_i = calibrate_synaptic_weights(config)
    assert j_e == j_i == 3.0  # S-p.19 target: both PSPs are 0.75mV in magnitude

    check = config["pair_model"]["calibration_check"]
    result = simulate_pair(
        e_spikes_a=[0.0], i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        duration_ms=check["duration_ms"], dt_ms=check["dt_ms"],
    )
    peak = float(result.v_a_mV.max())
    assert not np.isnan(peak)
    assert peak == pytest.approx(synapse["epsp_peak_mV"], rel=1e-3)

    _log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "block1_phase1_calibration",
        "J_E_mV": j_e, "J_I_mV": j_i, "simulated_EPSP_peak_mV": peak,
        "target_EPSP_peak_mV": synapse["epsp_peak_mV"], "nan_check_passed": True,
    })


def test_phase2_model_handles_realistic_pooled_load_against_real_config():
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    j_e, j_i = calibrate_synaptic_weights(config)

    rng = np.random.default_rng(config["seed"])
    duration_ms = 2000.0
    n_e = config["pair_model"]["inputs"]["n_excitatory"]
    rate_hz = config["pair_model"]["inputs"]["rate_shared_sweep_hz"]
    pooled_rate_hz = n_e * rate_hz
    n_events = rng.poisson(pooled_rate_hz * duration_ms / 1000.0)
    e_spikes_a = np.sort(rng.uniform(0, duration_ms - 1, n_events)).tolist()

    result = simulate_pair(
        e_spikes_a=e_spikes_a, i_spikes_a=[], e_spikes_b=[], i_spikes_b=[],
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=config["pair_model"]["simulation"]["dt_ms"],
    )

    nan_ok = not (
        np.isnan(result.v_a_mV).any()
        or np.isnan(result.v_b_mV).any()
        or np.isnan(result.i_syn_a_mV).any()
    )
    assert nan_ok
    assert len(result.spikes_b_ms) == 0  # cell B got no input

    cell_a_rate_hz = len(result.spikes_a_ms) / (duration_ms / 1000.0)
    _log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "block1_phase2_model_smoke_test",
        "n_pooled_e_events": int(n_events), "pooled_rate_hz": pooled_rate_hz,
        "cell_a_output_rate_hz": cell_a_rate_hz, "nan_check_passed": nan_ok,
    })
