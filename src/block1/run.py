"""Block 1's one entry point. Everything Fig. 1 needs is a subcommand here.

    python -m block1.run sweep          the paper-accurate pass, L=10,000 s, all 21 points
    python -m block1.run calibrate e_only | e_plus_i
    python -m block1.run check          qualitative trend checks against the paper

All 21 sweep points batch into ONE Brian2 simulation per time chunk -- the points do not
couple, so batching is exact and amortises Brian2's per-run cost across the whole grid.
Correlations accumulate per chunk rather than by concatenating every chunk's trace, which
would reinstate the memory footprint chunking exists to avoid.

Plots live in block1.plot; the machinery this drives lives in src/lib.
"""
import csv
import datetime
import hashlib
import sys
import time

import numpy as np

from analysis import StreamingCorrelation, spike_count_correlation
from block1.calibration import calibrate_from_config, calibrate_synaptic_weights
from block1.inputs import build_pair_inputs
from config import load_config
from lib.checks import check_increasing_trend
from lib.chunking import simulate_pairs_batch_chunked

CSV_PATH = "artifacts/block1_full_pass.csv"
CSV_FIELDS = [
    "phase", "p", "r_in", "n_e", "n_i", "rate_hz", "c", "r_out",
    "duration_s", "elapsed_s", "timestamp",
]


def _deterministic_seed_offset(phase: str, p: float, r_in: float, modulus: int = 100_000) -> int:
    """sha256, not hash(): Python randomises string hashing per process, so hash() would
    give a different grid on every run."""
    key = f"{phase}:{p!r}:{r_in!r}".encode()
    digest = hashlib.sha256(key).hexdigest()
    return int(digest, 16) % modulus


def build_sweep_points(config: dict) -> list[dict]:
    """The (phase, p, r_in) points making up Fig. 1B and Fig. 1E (E-only and E+I)."""
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
    # The E+I input rate is calibrated, not the paper's stated 20 Hz: measured, 20 Hz gives
    # ~12 Hz output, not the 5 Hz the SOM says it produces (config.yaml records the full
    # finding). Raise rather than silently fall back, which would put Fig. 1E's two curves
    # at different operating points -- the exact thing the recalibration exists to prevent.
    rate_e_plus_i = inputs_cfg["rate_e_plus_i_calibrated_hz"]
    if rate_e_plus_i is None:
        raise ValueError(
            "config.yaml: pair_model.inputs.rate_e_plus_i_calibrated_hz is null. Derive it "
            "first with `python -m block1.calibration e_plus_i`, then paste the value in. "
            "Falling back to the paper's stated 20 Hz would put Fig. 1E's E-only and E+I "
            "curves at ~5 Hz and ~12 Hz output respectively."
        )
    points += [
        dict(phase="fig1e_e_plus_i", p=p_fixed, r_in=r_in, n_e=n_e, n_i=n_i,
             rate_hz=rate_e_plus_i)
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
    command = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    config = load_config("config.yaml")

    if command == "calibrate":
        condition = sys.argv[2]
        rate = calibrate_from_config(condition, config)
        print(f"{condition}: input rate {rate:.5f} Hz gives ~5 Hz output at r_in = 0.\n"
              f"Put it in config.yaml and record the settings used.")
    elif command == "check":
        rows = [dict(r) for r in csv.DictReader(open(CSV_PATH))]
        for phase in ("fig1b", "fig1e_e_only", "fig1e_e_plus_i"):
            sub = sorted((r for r in rows if r["phase"] == phase),
                         key=lambda r: float(r["p"] if phase == "fig1b" else r["r_in"]))
            x = [float(r["p"] if phase == "fig1b" else r["r_in"]) for r in sub]
            ok = check_increasing_trend(x, [float(r["r_out"]) for r in sub],
                                        config["qualitative_checks"]["increasing_trend_min_correlation"])
            print(f"{phase}: r_out increases with the swept parameter -> {ok}")
    elif command == "sweep":
        rows = run_full_pass(config)
        write_csv(rows)
        print(f"wrote {len(rows)} rows to {CSV_PATH}")
    else:
        raise SystemExit(f"unknown command {command!r}; expected sweep, calibrate or check")


if __name__ == "__main__":
    main()
