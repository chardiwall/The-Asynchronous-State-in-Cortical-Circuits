"""Seam: run_p_sweep -- Fig. 1B's shared-fraction sweep (E-only, r_in=0)."""
import datetime
import json

import numpy as np
import pytest

from block1.calibration import calibrate_synaptic_weights
from block1.eval import check_increasing_trend
from block1.train import run_p_sweep, run_r_in_sweep
from config import load_config

COMMON_KWARGS = dict(
    # n_e=250 matches the real Fig. 1B scale (config.yaml pair_model.inputs.n_excitatory)
    # -- a much smaller n_e made cells fire too rarely over a short test window, giving a
    # zero-variance spike train and a correctly-NaN (not buggy) r_out at p=0.
    n_e=250, rate_hz=5.0, r_in=0.0,
    j_e_mV=3.0, j_i_mV=3.0,
    tau_m_ms=10.0, tau_s_ms=5.0, theta_mV=20.0, v_reset_mV=10.0, t_ref_ms=2.0,
    jitter_tau_ms=5.0,
    duration_ms=2000.0, dt_ms=0.05,
    bin_dt_ms=1.0, window_T_ms=50.0,
)


def test_returns_one_result_per_p_value_with_expected_keys():
    results = run_p_sweep(p_values=[0.0, 0.5], rng=np.random.default_rng(1), **COMMON_KWARGS)

    assert len(results) == 2
    for r in results:
        assert set(r.keys()) >= {"p", "c", "r_out"}
        assert not np.isnan(r["c"])
        assert not np.isnan(r["r_out"])


def test_p_one_gives_c_exactly_one():
    # p=1.0: every input is literally shared -> currents identical (Phase 4's finding).
    results = run_p_sweep(p_values=[1.0], rng=np.random.default_rng(2), **COMMON_KWARGS)
    assert results[0]["c"] == pytest.approx(1.0, abs=1e-9)


def test_c_is_higher_at_p_one_than_p_zero():
    # Directional sanity check (not a precise numeric match -- that's the later
    # paper-accurate pass): more sharing should mean more current correlation.
    results = run_p_sweep(
        p_values=[0.0, 1.0], rng=np.random.default_rng(3), **COMMON_KWARGS
    )
    c_at_p0 = results[0]["c"]
    c_at_p1 = results[1]["c"]
    assert c_at_p1 > c_at_p0


def test_r_in_sweep_returns_one_result_per_r_in_value():
    results = run_r_in_sweep(
        r_in_values=[0.0, 0.1], n_i=0, p=0.2, rng=np.random.default_rng(10),
        **{k: v for k, v in COMMON_KWARGS.items() if k != "r_in"},
    )
    assert len(results) == 2
    for r in results:
        assert set(r.keys()) >= {"p", "r_in", "c", "r_out"}
        assert not np.isnan(r["c"])
        assert not np.isnan(r["r_out"])


def test_r_in_sweep_shows_amplification_e_only():
    # M-Eq(1): c ~= p + N*r_in -- a substantial r_in should push c well above the r_in=0
    # baseline (which is just c~=p), confirming the amplification mechanism carries
    # through the r_in-sweep path too, not just the p-sweep path (Phase 5).
    results = run_r_in_sweep(
        r_in_values=[0.0, 0.05], n_i=0, p=0.2, rng=np.random.default_rng(11),
        **{k: v for k, v in COMMON_KWARGS.items() if k != "r_in"},
    )
    assert results[1]["c"] > results[0]["c"]


def test_fig1b_fast_exploratory_pass_against_real_config():
    # Phase 5, fast-exploratory scale (researcher-confirmed 2026-09-10, see PROGRESS.md):
    # L=200s (not the paper's L=10,000s) and a sparse p grid, to confirm the pipeline's
    # code/math correctness -- not a precise numeric match to Fig. 1B, which is deferred
    # to a later paper-accurate full pass.
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    analysis_cfg = config["analysis"]
    j_e, j_i = calibrate_synaptic_weights(config)

    p_values = [0.0, 0.1, 0.2, 0.3, 0.4]
    results = run_p_sweep(
        p_values=p_values,
        n_e=inputs_cfg["n_excitatory"], rate_hz=inputs_cfg["rate_shared_sweep_hz"], r_in=0.0,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        duration_ms=200_000.0, dt_ms=config["pair_model"]["simulation"]["dt_ms"],
        bin_dt_ms=analysis_cfg["bin_dt_ms"], window_T_ms=analysis_cfg["count_window_T_ms"],
        rng=np.random.default_rng(config["seed"]),
    )

    c_values = [r["c"] for r in results]
    r_out_values = [r["r_out"] for r in results]
    assert not any(np.isnan(c_values))
    assert not any(np.isnan(r_out_values))

    # Paper's stated qualitative result (docs/paper/01-postsynaptic-pair.md): "c and
    # r_out grow roughly linearly with p; both stay moderate (<=0.4-ish at p=0.4)".
    assert check_increasing_trend(p_values, c_values)
    assert check_increasing_trend(p_values, r_out_values)

    timestamp = datetime.datetime.now().isoformat()
    record = {
        "timestamp": timestamp, "phase": "block1_phase5_fig1b_fast_pass",
        "duration_ms": 200_000.0, "p_values": p_values,
        "c_values": c_values, "r_out_values": r_out_values,
    }
    with open("artifacts/metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    with open("artifacts/run.log", "a") as f:
        f.write(f"{timestamp} block1_phase5_fig1b_fast_pass: {record}\n")


def test_fig1e_fast_exploratory_pass_against_real_config():
    # Phase 6, fast-exploratory scale: L=200s (not L=10,000s), sparse r_in grid chosen
    # to capture the steep rise the docs describe ("r_out -> ~1 by r_in~=0.1" for E-only).
    # E-only rate uses the calibrated 4.0708Hz (ambiguity 1, resolved this session);
    # E+I uses the paper's stated 20Hz.
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    analysis_cfg = config["analysis"]
    p_fixed = config["pair_model"]["sweeps"]["p_fixed"]
    j_e, j_i = calibrate_synaptic_weights(config)

    r_in_values = [0.0, 0.01, 0.025, 0.05, 0.1]
    common = dict(
        p=p_fixed,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        duration_ms=200_000.0, dt_ms=config["pair_model"]["simulation"]["dt_ms"],
        bin_dt_ms=analysis_cfg["bin_dt_ms"], window_T_ms=analysis_cfg["count_window_T_ms"],
    )

    e_only = run_r_in_sweep(
        r_in_values=r_in_values, n_e=inputs_cfg["n_excitatory"], n_i=0,
        rate_hz=inputs_cfg["rate_e_only_calibrated_hz"],
        rng=np.random.default_rng(config["seed"]), **common,
    )
    e_plus_i = run_r_in_sweep(
        r_in_values=r_in_values, n_e=inputs_cfg["n_excitatory"],
        n_i=inputs_cfg["n_inhibitory"], rate_hz=inputs_cfg["rate_correlated_sweep_hz"],
        rng=np.random.default_rng(config["seed"] + 1), **common,
    )

    e_only_c = [r["c"] for r in e_only]
    e_only_r_out = [r["r_out"] for r in e_only]
    e_plus_i_c = [r["c"] for r in e_plus_i]
    e_plus_i_r_out = [r["r_out"] for r in e_plus_i]
    assert not any(np.isnan(v) for v in e_only_c + e_only_r_out + e_plus_i_c + e_plus_i_r_out)

    # Paper's stated qualitative results: E-only rises steeply; E+I is "strongly
    # suppressed" across the same r_in range -- E-only should end up well above E+I.
    assert check_increasing_trend(r_in_values, e_only_c)
    assert e_only_c[-1] > e_plus_i_c[-1]
    assert e_only_r_out[-1] > e_plus_i_r_out[-1]

    timestamp = datetime.datetime.now().isoformat()
    record = {
        "timestamp": timestamp, "phase": "block1_phase6_fig1e_fast_pass",
        "duration_ms": 200_000.0, "r_in_values": r_in_values,
        "e_only_c": e_only_c, "e_only_r_out": e_only_r_out,
        "e_plus_i_c": e_plus_i_c, "e_plus_i_r_out": e_plus_i_r_out,
    }
    with open("artifacts/metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    with open("artifacts/run.log", "a") as f:
        f.write(f"{timestamp} block1_phase6_fig1e_fast_pass: {record}\n")
