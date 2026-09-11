"""All four of Block 1's Fig. 1 plots.

1B/1E (plot_fig1b, plot_fig1e): read block1.full_pass's CSV, one combined panel each --
c as a dashed line, r_out as a marker-line ('-o-'), matching the paper's own Fig. 1B/E
layout (dashed c / open-circle r_out on one axis, not two separate panels). Palette:
dataviz skill's validated categorical slots 1 (blue, #2a78d6) and 2 (orange, #eb6834).

1C/1F (plot_fig1c, plot_fig1f): bespoke illustrative single-trial traces at p=0.2,
r_in=0.025 (the blue/black circle of Fig. 1E) -- purely visual, not a measurement, so no
TDD ceremony (same scope as this project's original traces.py). 3 stacked rows: input
raster (top), synaptic current (middle), membrane potential (bottom), scale bars instead
of full axes (matching the paper's own presentation). The raster shows each cell's
POOLED per-population arrival stream (2 rows: cell A, cell B) -- this project's model
operates on pooled arrival times, not individually-tracked synapses (block1/dataset.py),
so that's the coarsest-grained honest reading of "input raster" for this model; it's
less visually dense than the paper's own per-synapse raster but is exactly what drives
the simulation, not a separately-generated illustration.

Current-based model note: this project's "current" (i_syn_*_mV, calibration.py) is in mV
throughout, not nA -- the paper's own Fig. 1C/F scale bar is in nA because it illustrates
a literal current; the corresponding bar here is in mV, sized to the traces' own range.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from block1.calibration import calibrate_synaptic_weights
from block1.current_trace import synaptic_trace
from block1.dataset import build_pair_inputs, mother_train_pool
from block1.model import simulate_pair

BLUE = "#2a78d6"
ORANGE = "#eb6834"
N_DISPLAY_TRAINS = 30  # illustrative raster row count -- see _example_raster_trains


def load_full_pass_csv(path: str) -> dict[str, list[dict]]:
    rows_by_phase: dict[str, list[dict]] = {"fig1b": [], "fig1e_e_only": [], "fig1e_e_plus_i": []}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for key in ("p", "r_in", "c", "r_out", "duration_s", "elapsed_s"):
                row[key] = float(row[key])
            rows_by_phase[row["phase"]].append(row)
    for phase in rows_by_phase:
        rows_by_phase[phase].sort(key=lambda r: r["p"] if phase == "fig1b" else r["r_in"])
    return rows_by_phase


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


def plot_fig1b(rows: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_correlation_series(ax, "p", [(rows, BLUE, "")])
    ax.set_xlabel("Shared input fraction p")
    ax.set_title("Fig. 1B (E-only, r_in=0, L=10,000s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fig1e(e_only: list[dict], e_plus_i: list[dict], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    _plot_correlation_series(ax, "r_in", [(e_only, BLUE, "E only"), (e_plus_i, ORANGE, "E and I")])
    ax.set_xlabel("Input spike correlation r_in")
    ax.set_title("Fig. 1E (p=0.2, L=10,000s): E-only vs E+I")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _round_scale(value: float) -> float:
    """A visually clean scale-bar magnitude close to value (1/2/5 * 10^k)."""
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for m in (1, 2, 5, 10):
        candidate = m * 10 ** exponent
        if candidate >= value:
            return float(candidate)
    return float(10 ** (exponent + 1))


def _strip_axes_with_scale_bar(ax, y_range: float, y_unit: str, x_range_ms: float = 50.0) -> None:
    """Hides the box/ticks (matching the paper's scale-bar style) and draws an L-shaped
    scale bar in the bottom-right sized to the data: x_range_ms of time, a clean-rounded
    y_range of y_unit.
    """
    ax.axis("off")
    y_bar = _round_scale(y_range * 0.4)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    x0 = xlim[1] - x_range_ms
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + x_range_ms], [y0, y0], color="black", linewidth=1.5)
    ax.plot([x0, x0], [y0, y0 + y_bar], color="black", linewidth=1.5)
    ax.text(x0 + x_range_ms / 2, y0 - 0.03 * (ylim[1] - ylim[0]), f"{x_range_ms:.0f} ms",
            ha="center", va="top", fontsize=8)
    ax.text(x0 - 0.01 * (xlim[1] - xlim[0]), y0 + y_bar / 2, f"{y_bar:.2g} {y_unit}",
            ha="right", va="center", fontsize=8)


def _example_point(config: dict) -> tuple[float, float]:
    p = config["pair_model"]["sweeps"]["p_fixed"]
    r_in = config["pair_model"]["sweeps"]["r_in_fig1f_example"]
    return p, r_in


def _example_raster_trains(
    rate_hz: float, r_in: float, duration_ms: float, jitter_tau_ms: float,
    rng: np.random.Generator, n_display: int = N_DISPLAY_TRAINS,
) -> list[np.ndarray]:
    """A legible subsample of N_DISPLAY_TRAINS individual input trains, correlated at
    r_in (mother_train_pool, public), for the raster panel -- showing all ~250-470 real
    trains pooled into one row (this model's actual internal representation,
    block1/dataset.py) would render as a solid, illegible block at any duration long
    enough to show interesting dynamics; a real per-synapse raster figure subsamples for
    display too. Not the literal trains driving the current/V traces plotted alongside
    it (those come from the pooled construction) -- drawn at the same rate_hz/r_in so
    the displayed correlation structure matches what's actually driving the cell.
    """
    return mother_train_pool(rate_hz, r_in, n_display, duration_ms, jitter_tau_ms, rng)


def _plot_membrane_panel(ax, result) -> None:
    """The bottom row shared by Fig. 1C and 1F: both cells' V_m, scale-barred."""
    ax.plot(result.t_ms, result.v_a_mV, color="black")
    ax.plot(result.t_ms, result.v_b_mV, color="0.6")
    v_range = float(np.ptp(np.concatenate([result.v_a_mV, result.v_b_mV])))
    _strip_axes_with_scale_bar(ax, v_range, "mV")
    ax.set_title("membrane potential")


def plot_fig1c(config: dict, duration_ms: float, rng: np.random.Generator, out_path: str) -> None:
    """E-only: input raster (E arrivals onto each cell) / synaptic current / V_m."""
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    p, r_in = _example_point(config)
    j_e, j_i = calibrate_synaptic_weights(config)

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"], n_i=0, p=p, r_in=r_in,
        rate_hz=inputs_cfg["rate_e_only_calibrated_hz"], duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=[], e_spikes_b=inputs.e_spikes_b, i_spikes_b=[],
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms,
        dt_ms=config["pair_model"]["simulation"]["dt_ms"],
    )

    raster_trains = _example_raster_trains(
        inputs_cfg["rate_e_only_calibrated_hz"], r_in, duration_ms,
        inputs_cfg["mother_train"]["jitter_tau_ms"], rng,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].eventplot(raster_trains, colors="green", linelengths=0.8)
    axes[0].set_ylabel(f"{len(raster_trains)} example E inputs")
    axes[0].set_yticks([])
    axes[0].set_xlim(0, duration_ms)
    axes[0].set_title(f"Fig. 1C (E inputs only, p={p}, r_in={r_in}): input raster")

    axes[1].plot(result.t_ms, result.i_syn_a_mV, color="black")
    axes[1].plot(result.t_ms, result.i_syn_b_mV, color="0.6")
    i_range = float(np.ptp(np.concatenate([result.i_syn_a_mV, result.i_syn_b_mV])))
    _strip_axes_with_scale_bar(axes[1], i_range, "mV")
    axes[1].set_title("synaptic current")

    _plot_membrane_panel(axes[2], result)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fig1f(config: dict, duration_ms: float, rng: np.random.Generator, out_path: str) -> None:
    """E and I: input raster (E and I arrivals onto cell A) / E,I synaptic current
    (separately, cell A) / V_m -- matching Fig. 1C's layout.
    """
    neuron = config["pair_model"]["neuron"]
    synapse = config["pair_model"]["synapse"]
    inputs_cfg = config["pair_model"]["inputs"]
    dt_ms = config["pair_model"]["simulation"]["dt_ms"]
    p, r_in = _example_point(config)
    j_e, j_i = calibrate_synaptic_weights(config)

    inputs = build_pair_inputs(
        n_e=inputs_cfg["n_excitatory"], n_i=inputs_cfg["n_inhibitory"], p=p, r_in=r_in,
        rate_hz=inputs_cfg["rate_correlated_sweep_hz"], duration_ms=duration_ms,
        jitter_tau_ms=inputs_cfg["mother_train"]["jitter_tau_ms"], rng=rng,
    )
    result = simulate_pair(
        e_spikes_a=inputs.e_spikes_a, i_spikes_a=inputs.i_spikes_a,
        e_spikes_b=inputs.e_spikes_b, i_spikes_b=inputs.i_spikes_b,
        j_e_mV=j_e, j_i_mV=j_i,
        tau_m_ms=neuron["tau_m_ms"], tau_s_ms=synapse["tau_s_ms"],
        theta_mV=neuron["v_threshold_mV"], v_reset_mV=neuron["v_reset_mV"],
        t_ref_ms=neuron["t_refractory_ms"], duration_ms=duration_ms, dt_ms=dt_ms,
    )
    # Cell A's E/I current components shown separately (the paper's point: they excurse
    # together and cancel in the total) -- reuses synaptic_trace directly rather than
    # touching simulate_pair's tested interface.
    s_e_a, _ = synaptic_trace(inputs.e_spikes_a, duration_ms, dt_ms, synapse["tau_s_ms"])
    s_i_a, _ = synaptic_trace(inputs.i_spikes_a, duration_ms, dt_ms, synapse["tau_s_ms"])
    i_e_a = j_e * s_e_a
    i_i_a = -j_i * s_i_a

    n_each = N_DISPLAY_TRAINS // 2
    e_trains = _example_raster_trains(
        inputs_cfg["rate_correlated_sweep_hz"], r_in, duration_ms,
        inputs_cfg["mother_train"]["jitter_tau_ms"], rng, n_display=n_each,
    )
    i_trains = _example_raster_trains(
        inputs_cfg["rate_correlated_sweep_hz"], r_in, duration_ms,
        inputs_cfg["mother_train"]["jitter_tau_ms"], rng, n_display=n_each,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 7))
    axes[0].eventplot(list(i_trains) + list(e_trains),
                       colors=["red"] * n_each + ["green"] * n_each, linelengths=0.8)
    axes[0].axhline(n_each - 0.5, color="0.7", linewidth=1)
    axes[0].set_yticks([n_each / 2 - 0.5, n_each + n_each / 2 - 0.5])
    axes[0].set_yticklabels(["I -> cell A", "E -> cell A"])
    axes[0].set_xlim(0, duration_ms)
    axes[0].set_title(f"Fig. 1F (E and I inputs, p={p}, r_in={r_in}): input raster")

    axes[1].plot(result.t_ms, i_e_a, color="green", label="E current")
    axes[1].plot(result.t_ms, i_i_a, color="red", label="I current")
    i_range = float(np.ptp(np.concatenate([i_e_a, i_i_a])))
    _strip_axes_with_scale_bar(axes[1], i_range, "mV")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("synaptic current (E, I separately)")

    _plot_membrane_panel(axes[2], result)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
