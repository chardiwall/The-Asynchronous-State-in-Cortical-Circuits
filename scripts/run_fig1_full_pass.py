"""Paper-accurate full pass: Fig. 1B (p-sweep) and Fig. 1E (r_in-sweep, E-only + E+I) at
the paper's real L=10,000s, using simulate_pair_chunked (ADR 0002's deferred piece) to
keep peak memory bounded. Same sparse grid as the fast-exploratory pass (researcher-
confirmed 2026-09-10, see PROGRESS.md), now at full duration. Expected runtime: ~11
hours (15 points x ~43min/point at L=10,000s, per the Phase 3.6 benchmark).

Logs each point to artifacts/ incrementally (not batched at the end) so progress is
visible and partial results survive an interruption.
"""
import datetime
import json
import sys
import time

sys.path.insert(0, "src")

import numpy as np

from block1.calibration import calibrate_synaptic_weights
from block1.train import run_p_sweep, run_r_in_sweep
from config import load_config

CHUNK_DURATION_MS = 2_000_000.0  # 2000s per chunk -- 5 chunks per L=10,000s run


def log(record: dict) -> None:
    with open("artifacts/metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    with open("artifacts/run.log", "a") as f:
        f.write(f"{record['timestamp']} {record['phase']}: {record}\n")
    print(json.dumps(record))


def run_and_log_point(sweep_fn, point_kwargs, common, rng, phase_name, extra_fields):
    t0 = time.time()
    result = sweep_fn(rng=rng, chunk_duration_ms=CHUNK_DURATION_MS, **point_kwargs, **common)[0]
    elapsed_s = time.time() - t0
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": phase_name,
        "duration_ms": common["duration_ms"],
        "elapsed_s": elapsed_s,
        "c": result["c"],
        "r_out": result["r_out"],
        **extra_fields,
    }
    log(record)
    return result


def main():
    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    analysis_cfg = config["analysis"]
    p_fixed = config["pair_model"]["sweeps"]["p_fixed"]
    j_e, j_i = calibrate_synaptic_weights(config)
    rng = np.random.default_rng(config["seed"])

    duration_ms = config["pair_model"]["simulation"]["length_s"] * 1000.0  # 10,000,000ms
    dt_ms = config["pair_model"]["simulation"]["dt_ms"]

    common = dict(
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        duration_ms=duration_ms, dt_ms=dt_ms,
        bin_dt_ms=analysis_cfg["bin_dt_ms"], window_T_ms=analysis_cfg["count_window_T_ms"],
    )

    # Fig. 1B: p-sweep, E-only, r_in=0.
    fig1b_results = []
    for p in [0.0, 0.1, 0.2, 0.3, 0.4]:
        r = run_and_log_point(
            run_p_sweep, dict(p_values=[p], n_e=inputs_cfg["n_excitatory"],
                               rate_hz=inputs_cfg["rate_shared_sweep_hz"], r_in=0.0),
            common, rng, "block1_full_pass_fig1b", {"p": p},
        )
        fig1b_results.append(r)

    # Fig. 1E, E-only.
    fig1e_e_only = []
    for r_in in [0.0, 0.01, 0.025, 0.05, 0.1]:
        r = run_and_log_point(
            run_r_in_sweep, dict(r_in_values=[r_in], n_e=inputs_cfg["n_excitatory"],
                                  n_i=0, p=p_fixed, rate_hz=inputs_cfg["rate_e_only_calibrated_hz"]),
            common, rng, "block1_full_pass_fig1e_e_only", {"r_in": r_in},
        )
        fig1e_e_only.append(r)

    # Fig. 1E, E+I.
    fig1e_e_plus_i = []
    for r_in in [0.0, 0.01, 0.025, 0.05, 0.1]:
        r = run_and_log_point(
            run_r_in_sweep, dict(r_in_values=[r_in], n_e=inputs_cfg["n_excitatory"],
                                  n_i=inputs_cfg["n_inhibitory"], p=p_fixed,
                                  rate_hz=inputs_cfg["rate_correlated_sweep_hz"]),
            common, rng, "block1_full_pass_fig1e_e_plus_i", {"r_in": r_in},
        )
        fig1e_e_plus_i.append(r)

    log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": "block1_full_pass_summary",
        "fig1b_c": [r["c"] for r in fig1b_results],
        "fig1b_r_out": [r["r_out"] for r in fig1b_results],
        "fig1e_e_only_c": [r["c"] for r in fig1e_e_only],
        "fig1e_e_only_r_out": [r["r_out"] for r in fig1e_e_only],
        "fig1e_e_plus_i_c": [r["c"] for r in fig1e_e_plus_i],
        "fig1e_e_plus_i_r_out": [r["r_out"] for r in fig1e_e_plus_i],
    })


if __name__ == "__main__":
    main()
