"""Seam: stationary_correlation -- S-Eq(35-37), the shared measurement used identically
by blocks 1, 3 and 4 (07-analysis-methods.md). Cov(x,y) = (1/L)*sum((x-mean_x)(y-mean_y));
r = Cov(x,y)/sqrt(Cov(x,x)*Cov(y,y)) -- textbook Pearson correlation (library-first:
numpy.corrcoef), written to match the SOM's notation for auditability.
"""
import datetime
import json

import numpy as np
import pytest

from analysis import (
    StreamingCorrelation,
    spike_count_correlation,
    stationary_correlation,
    windowed_rate,
)
from block1.calibration import calibrate_synaptic_weights
from block1.inputs import build_pair_inputs
from block1.model import simulate_pair
from config import load_config, output_path


def _log(record: dict) -> None:
    config = load_config("config.yaml")
    with open(output_path(config, "metrics_jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(output_path(config, "run_log"), "a") as f:
        f.write(f"{record['timestamp']} {record['phase']}: {record}\n")


def test_perfectly_correlated_signals_give_r_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x  # exact linear relationship, positive slope
    assert stationary_correlation(x, y) == pytest.approx(1.0)


def test_perfectly_anticorrelated_signals_give_r_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = -2.0 * x + 10.0  # exact linear relationship, negative slope
    assert stationary_correlation(x, y) == pytest.approx(-1.0)


def test_orthogonal_signals_give_r_zero():
    # Hand-verified: mean(x)=mean(y)=0, Cov(x,y)=mean([1,-1,-1,1])=0 exactly.
    x = np.array([1.0, -1.0, 1.0, -1.0])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    assert stationary_correlation(x, y) == pytest.approx(0.0, abs=1e-9)


def test_streaming_correlation_matches_stationary_correlation_on_one_chunk():
    rng = np.random.default_rng(5)
    x = rng.normal(size=1000)
    y = 0.7 * x + rng.normal(scale=0.5, size=1000)

    acc = StreamingCorrelation()
    acc.update(x, y)
    assert acc.correlation() == pytest.approx(stationary_correlation(x, y), abs=1e-9)


def test_streaming_correlation_matches_when_split_into_chunks():
    # The point of StreamingCorrelation: chunking the input must not change the result
    # (block1.run streams chunks instead of holding the full L=10,000s array).
    rng = np.random.default_rng(6)
    x = rng.normal(loc=3.0, scale=2.0, size=10_000)
    y = -0.4 * x + rng.normal(scale=1.5, size=10_000)
    expected = stationary_correlation(x, y)

    acc = StreamingCorrelation()
    chunk_size = 777  # deliberately doesn't divide 10,000 evenly
    for start in range(0, len(x), chunk_size):
        acc.update(x[start:start + chunk_size], y[start:start + chunk_size])

    assert acc.correlation() == pytest.approx(expected, abs=1e-9)


def test_windowed_rate_matches_hand_computed_sliding_sum():
    # S-Eq 34: n_i(t;T) = K_T(t)*s_i(t), K_T = 1/T on (t,t+T) -- a FORWARD-looking
    # sliding sum of spike counts, normalised to spikes/s. Hand-computed independently:
    # spikes at [0.5,2.5,3.5,5.2,5.7]ms binned at dt=1ms give per-bin counts
    # [1,0,1,1,0,2,0] over 7 bins; T=3ms (window_samples=3) sliding sums are
    # [2,2,2,3,2] (5 valid windows: N-window+1=7-3+1=5), each divided by T_seconds=0.003s.
    spike_times_ms = [0.5, 2.5, 3.5, 5.2, 5.7]
    rate = windowed_rate(spike_times_ms, duration_ms=7.0, bin_dt_ms=1.0, window_T_ms=3.0)

    expected_sums = np.array([2, 2, 2, 3, 2])
    expected = expected_sums / 0.003  # spikes/s
    assert rate == pytest.approx(expected, rel=1e-9)


def test_identical_spike_trains_give_r_out_one():
    # r_out is the pair's spike-count correlation (S-Eq 34-37 combined) -- identical
    # trains must give r_out=1 exactly, the cleanest possible sanity check.
    rng = np.random.default_rng(1)
    spikes = np.sort(rng.uniform(0, 10_000.0, 500)).tolist()

    r_out = spike_count_correlation(
        spikes, spikes, duration_ms=10_000.0, bin_dt_ms=1.0, window_T_ms=50.0
    )
    assert r_out == pytest.approx(1.0, abs=1e-9)


def test_independent_spike_trains_give_r_out_near_zero():
    rng = np.random.default_rng(2)
    duration_ms = 100_000.0  # 100s, long enough to average out sampling noise
    spikes_a = np.sort(rng.uniform(0, duration_ms, 500)).tolist()
    spikes_b = np.sort(rng.uniform(0, duration_ms, 500)).tolist()

    r_out = spike_count_correlation(
        spikes_a, spikes_b, duration_ms=duration_ms, bin_dt_ms=1.0, window_T_ms=50.0
    )
    assert r_out == pytest.approx(0.0, abs=0.1)


def test_fully_shared_input_gives_strongly_positive_c_and_r_out_against_real_config():
    # Wires Phase 4's measurement to the real Phase 2/3 pipeline: p=1.0 means every
    # input is literally identical between the two cells (build_pair_inputs), so their
    # currents must be perfectly correlated (c=1 exactly) and their output spiking
    # strongly positively correlated too -- a directional sanity check ahead of Phase
    # 5/6's actual Fig. 1 numeric targets.
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    analysis_cfg = config["analysis"]
    j_e, j_i = calibrate_synaptic_weights(config)

    rng = np.random.default_rng(config["seed"])
    duration_ms = 5000.0
    inputs = build_pair_inputs(
        n_e=config["pair_model"]["inputs"]["n_excitatory"], n_i=0,
        p=1.0, r_in=0.0, rate_hz=config["pair_model"]["inputs"]["rate_shared_sweep_hz"],
        duration_ms=duration_ms,
        jitter_tau_ms=config["pair_model"]["inputs"]["mother_train"]["jitter_tau_ms"],
        rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=config["pair_model"]["simulation"]["dt_ms"],
    )

    c = stationary_correlation(result.i_syn_a_mV, result.i_syn_b_mV)
    r_out = spike_count_correlation(
        result.spikes_a_ms, result.spikes_b_ms, duration_ms,
        analysis_cfg["bin_dt_ms"], analysis_cfg["count_window_T_ms"],
    )

    assert not np.isnan(c)
    assert not np.isnan(r_out)
    assert c == pytest.approx(1.0, abs=1e-9)  # p=1.0: currents are literally identical
    assert r_out > 0.5  # strongly positive; not necessarily exactly 1 (spiking is nonlinear)

    _log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "block1_phase4_measurement_pipeline",
        "p": 1.0, "r_in": 0.0, "duration_ms": duration_ms,
        "c": c, "r_out": r_out, "nan_check_passed": True,
    })
