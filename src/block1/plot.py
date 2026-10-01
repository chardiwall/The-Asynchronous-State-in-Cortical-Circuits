"""Fig. 1's four panels. Usage: python -m block1.plot [trace_duration_ms]

1B and 1E are sweep curves from the CSV block1.run writes: c dashed, r_out as a marker line,
one combined panel each. 1C and 1F are illustrative single-trial traces at the example point
-- raster, synaptic current, membrane potential, scale bars instead of axes, purely visual.
1F splits the current into E and I components: they excurse together and cancel, which is
the panel's point. The bar is in mV, not the paper's nA, since this model is current-based
with V relative to rest.
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from block1.calibration import calibrate_synaptic_weights
from block1.inputs import build_pair_inputs, mother_train_pool
from block1.model import simulate_pair
from config import load_config, output_path
from lib.plotting import BLUE, ORANGE, scale_bar
from lib.psc import synaptic_trace

N_DISPLAY_TRAINS = 30  # illustrative raster row count -- see _raster_trains

def load_full_pass_csv(path: str) -> dict[str, list[dict]]:
    """The sweep CSV grouped by phase, each group sorted along its own swept parameter."""
    by_phase: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(open(path, newline="")):
        by_phase[row["phase"]].append(
            {**row, **{k: float(row[k]) for k in ("p", "r_in", "c", "r_out")}})
    for phase, rows in by_phase.items():
        rows.sort(key=lambda r: r["p"] if phase == "fig1b" else r["r_in"])
    return by_phase


def _plot_correlation_series(ax, x_key: str, series: list[tuple[list[dict], str, str]]) -> None:
    """Draws c (dashed) and r_out ('-o-') for each (rows, color, label) series on one
    axis -- the shared shape of Fig. 1B (one series) and Fig. 1E (two: E-only, E+I).
    """
    for rows, color, label in series:
        x = [r[x_key] for r in rows]
        c = [r["c"] for r in rows]
        r_out = [r["r_out"] for r in rows]
        c_label = f"{label}: c" if label else "c"
        r_out_label = f"{label}: r_out" if label else "r_out"
        ax.plot(x, c, "--", color=color, linewidth=2, label=c_label)
        ax.plot(x, r_out, "-o", color=color, linewidth=2, markersize=7, label=r_out_label)
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Correlation")
    ax.legend(frameon=False, fontsize=8)


def plot_sweep(series, x_key: str, x_label: str, title: str, out_path: str) -> None:
    """Either sweep panel: 1B is one series against p, 1E is two against r_in."""
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_correlation_series(ax, x_key, series)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _raster_trains(rate_hz, r_in, duration_ms, jitter_tau_ms, rng, n_display=N_DISPLAY_TRAINS):
    """A legible subsample of individual input trains correlated at r_in. The real
    ~250-470 trains are pooled into one arrival stream internally (block1/inputs.py) and
    would render as a solid block; a per-synapse raster figure subsamples for display too.
    Drawn at the same rate_hz/r_in as the trains actually driving the cell, so the
    displayed correlation structure matches, but these are not literally those trains.
    """
    return mother_train_pool(rate_hz, r_in, n_display, duration_ms, jitter_tau_ms, rng)


def _run_example(config: dict, duration_ms: float, rng, with_inhibition: bool):
    """One illustrative trial at the Fig. 1E example point. The two conditions differ only
    in whether I inputs exist and which calibrated rate applies, so they share this path."""
    pair = config["pair_model"]
    neuron, synapse, inputs_cfg = pair["neuron"], pair["synapse"], pair["inputs"]
    p = pair["sweeps"]["p_fixed"]
    r_in = pair["sweeps"]["r_in_fig1f_example"]
    # Both conditions use a calibrated input rate so they share an operating point; see
    # config.yaml's note on rate_e_plus_i_calibrated_hz for why the paper's stated 20 Hz
    # is not used for the E+I condition.
    rate_hz = (inputs_cfg["rate_e_plus_i_calibrated_hz"] if with_inhibition
               else inputs_cfg["rate_e_only_calibrated_hz"])
    if rate_hz is None:
        raise ValueError("config.yaml: rate_e_plus_i_calibrated_hz is null -- derive it with "
                         "`python -m block1.calibration e_plus_i` before plotting Fig. 1F.")
    j_e, j_i = calibrate_synaptic_weights(config)

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"],
        n_i=inputs_cfg["n_inhibitory"] if with_inhibition else 0,
        p=p, r_in=r_in, rate_hz=rate_hz, duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=pair["simulation"]["dt_ms"],
    )
    return inputs, result, (j_e, j_i), (p, r_in, rate_hz)


def _membrane_panel(ax, result) -> None:
    """1C and 1F's shared bottom row: both cells' V_m."""
    ax.plot(result.t_ms, result.v_a_mV, color="black")
    ax.plot(result.t_ms, result.v_b_mV, color="0.6")
    scale_bar(ax, float(np.ptp(np.concatenate([result.v_a_mV, result.v_b_mV]))), "mV")
    ax.set_title("membrane potential")


def plot_fig1c(config: dict, duration_ms: float, rng, out_path: str) -> None:
    """E only: input raster / total synaptic current / V_m."""
    inputs, result, _, (p, r_in, rate_hz) = _run_example(config, duration_ms, rng, False)
    jitter = config["pair_model"]["inputs"]["mother_train"]["jitter_tau_ms"]
    trains = _raster_trains(rate_hz, r_in, duration_ms, jitter, rng)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].eventplot(trains, colors="green", linelengths=0.8)
    axes[0].set_ylabel(f"{len(trains)} example E inputs")
    axes[0].set_yticks([])
    axes[0].set_xlim(0, duration_ms)
    axes[0].set_title(f"Fig. 1C (E inputs only, p={p}, r_in={r_in}): input raster")

    axes[1].plot(result.t_ms, result.i_syn_a_mV, color="black")
    axes[1].plot(result.t_ms, result.i_syn_b_mV, color="0.6")
    scale_bar(axes[1], float(np.ptp(np.concatenate([result.i_syn_a_mV, result.i_syn_b_mV]))), "mV")
    axes[1].set_title("synaptic current")

    _membrane_panel(axes[2], result)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fig1f(config: dict, duration_ms: float, rng, out_path: str) -> None:
    """E and I: raster of both input populations / cell A's E and I currents shown
    SEPARATELY (they excurse together and cancel -- the panel's point) / V_m.
    """
    inputs, result, (j_e, j_i), (p, r_in, rate_hz) = _run_example(config, duration_ms, rng, True)
    pair = config["pair_model"]
    tau_s, dt_ms = pair["synapse"]["tau_s_ms"], pair["simulation"]["dt_ms"]
    jitter = pair["inputs"]["mother_train"]["jitter_tau_ms"]

    # Components via synaptic_trace directly, rather than widening simulate_pair's
    # tested interface just to expose them for one figure.
    i_e_a = j_e * synaptic_trace(inputs.e_spikes_a, duration_ms, dt_ms, tau_s)[0]
    i_i_a = -j_i * synaptic_trace(inputs.i_spikes_a, duration_ms, dt_ms, tau_s)[0]

    half = N_DISPLAY_TRAINS // 2
    e_trains = _raster_trains(rate_hz, r_in, duration_ms, jitter, rng, half)
    i_trains = _raster_trains(rate_hz, r_in, duration_ms, jitter, rng, half)

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].eventplot(list(i_trains) + list(e_trains),
                       colors=["red"] * half + ["green"] * half, linelengths=0.8)
    axes[0].axhline(half - 0.5, color="0.7", linewidth=1)
    axes[0].set_yticks([half / 2 - 0.5, half + half / 2 - 0.5])
    axes[0].set_yticklabels(["I -> cell A", "E -> cell A"])
    axes[0].set_xlim(0, duration_ms)
    axes[0].set_title(f"Fig. 1F (E and I inputs, p={p}, r_in={r_in}): input raster")

    axes[1].plot(result.t_ms, i_e_a, color="green", label="E current")
    axes[1].plot(result.t_ms, i_i_a, color="red", label="I current")
    scale_bar(axes[1], float(np.ptp(np.concatenate([i_e_a, i_i_a]))), "mV")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("synaptic current (E, I separately)")

    _membrane_panel(axes[2], result)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    config = load_config("config.yaml")
    rows = load_full_pass_csv(output_path(config, "block1_sweep_csv"))
    plot_sweep([(rows["fig1b"], BLUE, "")], "p", "Shared input fraction p",
               "Fig. 1B (E-only, r_in=0, L=10,000s)", output_path(config, "block1_fig1b"))
    plot_sweep([(rows["fig1e_e_only"], BLUE, "E only"),
                (rows["fig1e_e_plus_i"], ORANGE, "E and I")],
               "r_in", "Input spike correlation r_in",
               "Fig. 1E (p=0.2, L=10,000s): E-only vs E+I", output_path(config, "block1_fig1e"))

    # A few hundred ms of one trial -- these panels show shape, not a measurement.
    duration_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0
    rng = np.random.default_rng(config["seed"])
    plot_fig1c(config, duration_ms, rng, output_path(config, "block1_fig1c"))
    plot_fig1f(config, duration_ms, rng, output_path(config, "block1_fig1f"))
    print(f"wrote fig1b/c/e/f.png to {config['paths']['artifacts']}/")
