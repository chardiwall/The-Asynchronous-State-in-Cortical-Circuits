"""Block 1's main script: the paper-accurate full pass (L=10,000s) across all three Fig. 1
sweeps -- 1B's p (E-only, r_in=0), 1E's r_in for E-only and for E+I. All 15 points are
batched into ONE Brian2 simulation per time chunk (batched_model.simulate_pairs_batch_chunked)
instead of one Brian2 run per point: the points don't couple to each other, so batching
is exact, and it turns the previous run's dominant cost (many separate per-point Brian2
b2.run() calls, see PROGRESS.md's Phase 3.6 benchmark) into one amortized cost, giving a
GPU backend real per-timestep width to parallelize over.

Correlations are accumulated per chunk (analysis.StreamingCorrelation) rather than by
concatenating every chunk's full current trace and correlating at the end -- holding all
chunks' (n_samples, 15) arrays simultaneously would reinstate the full-duration memory
footprint chunking exists to avoid.

This replaced a subprocess-per-point architecture that existed purely to survive memory
pressure on a much smaller machine; on the DGX (121GB RAM) that isolation is unnecessary,
so this script runs as one ordinary process and writes a single CSV at the end.

Usage: python -m block1.full_pass
"""
import csv
import datetime
import hashlib
import time

import numpy as np

from analysis import StreamingCorrelation, spike_count_correlation
from block1.batched_model import simulate_pairs_batch_chunked
from block1.calibration import calibrate_synaptic_weights
from block1.dataset import build_pair_inputs
from config import load_config

CSV_PATH = "artifacts/block1_full_pass.csv"
CSV_FIELDS = [
    "phase", "p", "r_in", "n_e", "n_i", "rate_hz", "c", "r_out",
    "duration_s", "elapsed_s", "timestamp",
]


def _deterministic_seed_offset(phase: str, p: float, r_in: float, modulus: int = 100_000) -> int:
    """sha256 of a canonical string encoding -- reproducible across processes/machines,
    unlike Python's hash() of strings (randomized per-process via PYTHONHASHSEED).
    """
    key = f"{phase}:{p!r}:{r_in!r}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % modulus


def build_sweep_points(config: dict) -> list[dict]:
    """The 15 (phase, p, r_in) points making up Fig. 1B + Fig. 1E (E-only, E+I)."""
    inputs_cfg = config["pair_model"]["inputs"]
    sweeps = config["pair_model"]["sweeps"]
    n_e = inputs_cfg["n_excitatory"]
    n_i = inputs_cfg["n_inhibitory"]
    p_fixed = sweeps["p_fixed"]

    points = [
        dict(phase="fig1b", p=p, r_in=0.0, n_e=n_e, n_i=0, rate_hz=inputs_cfg["rate_shared_sweep_hz"])
        for p in sweeps["shared_fraction_grid"]
    ]
    points += [
        dict(phase="fig1e_e_only", p=p_fixed, r_in=r_in, n_e=n_e, n_i=0,
             rate_hz=inputs_cfg["rate_e_only_calibrated_hz"])
        for r_in in sweeps["r_in_grid"]
    ]
    points += [
        dict(phase="fig1e_e_plus_i", p=p_fixed, r_in=r_in, n_e=n_e, n_i=n_i,
             rate_hz=inputs_cfg["rate_correlated_sweep_hz"])
        for r_in in sweeps["r_in_grid"]
    ]
    return points


def run_full_pass(config: dict, points: list[dict] | None = None) -> list[dict]:
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    sim_cfg = config["pair_model"]["simulation"]
    analysis_cfg = config["analysis"]
    jitter_tau_ms = config["pair_model"]["inputs"]["mother_train"]["jitter_tau_ms"]
    j_e, j_i = calibrate_synaptic_weights(config)

    points = points if points is not None else build_sweep_points(config)
    n = len(points)
    duration_ms = sim_cfg["length_s"] * 1000.0
    chunk_duration_ms = sim_cfg["chunk_duration_s"] * 1000.0
    dt_ms = sim_cfg["dt_ms"]

    rngs = [
        np.random.default_rng(config["seed"] + _deterministic_seed_offset(pt["phase"], pt["p"], pt["r_in"]))
        for pt in points
    ]

    def input_provider(start_ms: float, end_ms: float):
        chunk_ms = end_ms - start_ms
        e_a, i_a, e_b, i_b = [], [], [], []
        for pt, rng in zip(points, rngs):
            inputs = build_pair_inputs(
                n_e=pt["n_e"], n_i=pt["n_i"], p=pt["p"], r_in=pt["r_in"], rate_hz=pt["rate_hz"],
                duration_ms=chunk_ms, jitter_tau_ms=jitter_tau_ms, rng=rng,
            )
            e_a.append(inputs.e_spikes_a); i_a.append(inputs.i_spikes_a)
            e_b.append(inputs.e_spikes_b); i_b.append(inputs.i_spikes_b)
        return e_a, i_a, e_b, i_b

    correlations = [StreamingCorrelation() for _ in range(n)]
    spikes_a_chunks = [[] for _ in range(n)]
    spikes_b_chunks = [[] for _ in range(n)]

    t0 = time.time()
    for start_ms, end_ms, result in simulate_pairs_batch_chunked(
        n=n, input_provider=input_provider,
        j_e_mV=j_e, j_i_mV=j_i, tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"],
        total_duration_ms=duration_ms, chunk_duration_ms=chunk_duration_ms, dt_ms=dt_ms,
    ):
        for k in range(n):
            correlations[k].update(result.i_syn_a_mV[:, k], result.i_syn_b_mV[:, k])
            spikes_a_chunks[k].append(result.spikes_a_ms[k] + start_ms)
            spikes_b_chunks[k].append(result.spikes_b_ms[k] + start_ms)
        print(f"chunk [{start_ms/1000:.0f}s, {end_ms/1000:.0f}s) done, "
              f"elapsed {time.time() - t0:.0f}s", flush=True)

    elapsed_s = time.time() - t0

    rows = []
    for k, pt in enumerate(points):
        spikes_a = np.concatenate(spikes_a_chunks[k]) if spikes_a_chunks[k] else np.array([])
        spikes_b = np.concatenate(spikes_b_chunks[k]) if spikes_b_chunks[k] else np.array([])
        r_out = spike_count_correlation(
            spikes_a, spikes_b, duration_ms, analysis_cfg["bin_dt_ms"], analysis_cfg["count_window_T_ms"]
        )
        rows.append({
            "phase": pt["phase"], "p": pt["p"], "r_in": pt["r_in"],
            "n_e": pt["n_e"], "n_i": pt["n_i"], "rate_hz": pt["rate_hz"],
            "c": correlations[k].correlation(), "r_out": r_out,
            "duration_s": sim_cfg["length_s"], "elapsed_s": elapsed_s,
            "timestamp": datetime.datetime.now().isoformat(),
        })
    return rows


def write_csv(rows: list[dict], path: str = CSV_PATH) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = load_config("config.yaml")
    rows = run_full_pass(config)
    write_csv(rows)
    print(f"wrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()
