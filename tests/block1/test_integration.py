"""End-to-end checks against the real config.yaml (not a fixture) -- this is the
committed, reproducible source for the "ran end-to-end, NaN-checked" claims. Also
appends to artifacts/ so there's a log trail of real runs.
"""
import datetime
import json

import numpy as np
import pytest

from block1.calibration import calibrate_synaptic_weights
from block1.inputs import build_pair_inputs
from block1.model import simulate_pair
from config import load_config


def _zoh_bias_tol(dt_ms: float, tau_s_ms: float, safety_factor: float = 3.0) -> float:
    """Tolerance for TimedArray's zero-order-hold bias: O(dt/tau_s)."""
    return safety_factor * dt_ms / tau_s_ms


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
    tol = _zoh_bias_tol(check["dt_ms"], synapse["tau_s_ms"])
    assert peak == pytest.approx(synapse["epsp_peak_mV"], rel=tol)

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


def test_phase3_pipeline_handles_hundreds_of_thousands_of_pooled_events():
    # Phase 3.6: sanity check at a scale the OLD per-event Brian2 mechanism could not
    # have built at all. Full real N_E/N_I/rate from config.yaml, at Fig.
    # 1F's exact (p, r_in) operating point, but a reduced duration (10s not 10,000s) so
    # this stays fast enough for the regular suite -- event *count* is what's being
    # stress-tested here, not wall-clock time at full duration (that's a separate,
    # deliberately-not-automated benchmark. The real-scale timing this smoke test's
    # scale was chosen from: 9.4M events / 1000s took 2.2s to
    # generate and 258.8s to simulate in Brian2 -- input generation is no longer the
    # bottleneck the precomputed-trace rework fixed; Brian2's own per-timestep cost now
    # dominates, which is the "trivially parallel over (p,r_in) grid points" cost the
    # paper's own figure protocol already anticipated, not a new blocker).
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    j_e, j_i = calibrate_synaptic_weights(config)

    rng = np.random.default_rng(config["seed"])
    duration_ms = 10_000.0  # 10s
    rate_hz = inputs_cfg["rate_correlated_sweep_hz"]  # 20 Hz, Fig. 1C/E/F
    p = config["pair_model"]["sweeps"]["p_fixed"]  # 0.2
    r_in = config["pair_model"]["sweeps"]["r_in_fig1f_example"]

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"], n_i=inputs_cfg["n_inhibitory"],
        p=p, r_in=r_in, rate_hz=rate_hz, duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    total_events = (
        len(inputs.e_spikes_a) + len(inputs.i_spikes_a)
        + len(inputs.e_spikes_b) + len(inputs.i_spikes_b)
    )
    assert total_events > 100_000  # confirms this is genuinely stress-testing scale

    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=config["pair_model"]["simulation"]["dt_ms"],
    )

    nan_ok = not (np.isnan(result.v_a_mV).any() or np.isnan(result.v_b_mV).any())
    assert nan_ok
    rate_a_hz = len(result.spikes_a_ms) / (duration_ms / 1000.0)
    rate_b_hz = len(result.spikes_b_ms) / (duration_ms / 1000.0)
    assert 0 < rate_a_hz < 200  # plausibility bound, not a Fig. 1 numeric match (Phase 6)
    assert 0 < rate_b_hz < 200

    _log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "block1_phase3_scale_smoke_test",
        "total_pooled_events": total_events, "duration_ms": duration_ms,
        "p": p, "r_in": r_in, "rate_a_hz": rate_a_hz, "rate_b_hz": rate_b_hz,
        "nan_check_passed": nan_ok,
    })
