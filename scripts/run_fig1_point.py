"""Computes exactly ONE Fig. 1 sweep point and logs it to artifacts/. Invoked as its own
subprocess (see run_fig1_full_pass.py) so the OS fully reclaims memory between points --
Python/numpy don't reliably release freed memory back to the OS within a long-lived
process, so a full ~15-point sweep run in one process lets RSS creep upward across
points until it triggers an OOM kill (found 2026-09-10/11: the process was killed
partway into the Fig. 1E E+I sweep, after 10/15 points had already completed correctly).

Usage: python scripts/run_fig1_point.py <phase> <p_or_r_in>
  phase: fig1b | fig1e_e_only | fig1e_e_plus_i
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

CHUNK_DURATION_MS = 2_000_000.0  # 2000s per chunk


def log(record: dict) -> None:
    with open("artifacts/metrics.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    with open("artifacts/run.log", "a") as f:
        f.write(f"{record['timestamp']} {record['phase']}: {record}\n")


def main():
    phase, value_str = sys.argv[1], sys.argv[2]
    value = float(value_str)

    config = load_config("config.yaml")
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    analysis_cfg = config["analysis"]
    p_fixed = config["pair_model"]["sweeps"]["p_fixed"]
    j_e, j_i = calibrate_synaptic_weights(config)
    # Seed depends on (phase, value) so each point is reproducible and independent of
    # ordering/how many other points ran before it in this or another process.
    rng = np.random.default_rng(config["seed"] + hash((phase, value)) % 100_000)

    duration_ms = config["pair_model"]["simulation"]["length_s"] * 1000.0
    common = dict(
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"],
        duration_ms=duration_ms, dt_ms=config["pair_model"]["simulation"]["dt_ms"],
        bin_dt_ms=analysis_cfg["bin_dt_ms"], window_T_ms=analysis_cfg["count_window_T_ms"],
        chunk_duration_ms=CHUNK_DURATION_MS,
    )

    t0 = time.time()
    if phase == "fig1b":
        result = run_p_sweep(
            p_values=[value], n_e=inputs_cfg["n_excitatory"],
            rate_hz=inputs_cfg["rate_shared_sweep_hz"], r_in=0.0, rng=rng, **common,
        )[0]
        extra = {"p": value}
    elif phase == "fig1e_e_only":
        result = run_r_in_sweep(
            r_in_values=[value], n_e=inputs_cfg["n_excitatory"], n_i=0, p=p_fixed,
            rate_hz=inputs_cfg["rate_e_only_calibrated_hz"], rng=rng, **common,
        )[0]
        extra = {"r_in": value}
    elif phase == "fig1e_e_plus_i":
        result = run_r_in_sweep(
            r_in_values=[value], n_e=inputs_cfg["n_excitatory"],
            n_i=inputs_cfg["n_inhibitory"], p=p_fixed,
            rate_hz=inputs_cfg["rate_correlated_sweep_hz"], rng=rng, **common,
        )[0]
        extra = {"r_in": value}
    else:
        raise ValueError(f"unknown phase {phase!r}")

    log({
        "timestamp": datetime.datetime.now().isoformat(),
        "phase": f"block1_full_pass_{phase}",
        "duration_ms": duration_ms,
        "elapsed_s": time.time() - t0,
        "c": result["c"], "r_out": result["r_out"],
        **extra,
    })


if __name__ == "__main__":
    main()
