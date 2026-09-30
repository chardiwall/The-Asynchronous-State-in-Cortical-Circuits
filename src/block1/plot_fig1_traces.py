"""Fig. 1C and 1F's illustrative single-trial traces at p=0.2, r_in=0.025 (the black and
blue circles of Fig. 1E). Purely visual, not a measurement: each runs one short fresh
simulation and draws three stacked rows -- input raster, synaptic current, membrane
potential -- with scale bars instead of full axes, matching the paper's presentation.
1C is the E-only condition, 1F adds I and splits the current row into its E and I
components, which is the panel's whole point: they excurse together and cancel.

The sweep curves 1B/1E live in plot_fig1.py; the two answer different questions.

Current-based model note: this project's current (i_syn_*_mV, calibration.py) is in mV,
not nA -- the paper's own 1C/1F scale bar is in nA because it illustrates a literal
current; the bar here is in mV, sized to the traces' own range.

Usage: python -m block1.plot_fig1_traces [duration_ms]
"""
import matplotlib.pyplot as plt
import numpy as np

from block1.calibration import calibrate_synaptic_weights
from block1.current_trace import synaptic_trace
from block1.dataset import build_pair_inputs, mother_train_pool
from block1.model import simulate_pair

N_DISPLAY_TRAINS = 30  # illustrative raster row count -- see _raster_trains


def _round_scale(value: float) -> float:
    """A visually clean scale-bar magnitude close to value (1/2/5 * 10^k)."""
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for m in (1, 2, 5, 10):
        if m * 10 ** exponent >= value:
            return float(m * 10 ** exponent)
    return float(10 ** (exponent + 1))


def _scale_bar(ax, y_range: float, y_unit: str, x_range_ms: float = 50.0) -> None:
    """Hides the box and ticks (the paper's style) and draws an L-shaped scale bar in the
    bottom-right, sized to the data: x_range_ms of time by a clean-rounded y_range.
    """
    ax.axis("off")
    y_bar = _round_scale(y_range * 0.4)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[1] - x_range_ms
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + x_range_ms], [y0, y0], color="black", linewidth=1.5)
    ax.plot([x0, x0], [y0, y0 + y_bar], color="black", linewidth=1.5)
    ax.text(x0 + x_range_ms / 2, y0 - 0.03 * (ylim[1] - ylim[0]), f"{x_range_ms:.0f} ms",
            ha="center", va="top", fontsize=8)
    ax.text(x0 - 0.01 * (xlim[1] - xlim[0]), y0 + y_bar / 2, f"{y_bar:.2g} {y_unit}",
            ha="right", va="center", fontsize=8)


def _raster_trains(rate_hz, r_in, duration_ms, jitter_tau_ms, rng, n_display=N_DISPLAY_TRAINS):
    """A legible subsample of individual input trains correlated at r_in. The real
    ~250-470 trains are pooled into one arrival stream internally (block1/dataset.py) and
    would render as a solid block; a per-synapse raster figure subsamples for display too.
    Drawn at the same rate_hz/r_in as the trains actually driving the cell, so the
    displayed correlation structure matches, but these are not literally those trains.
    """
    return mother_train_pool(rate_hz, r_in, n_display, duration_ms, jitter_tau_ms, rng)


def _run_example(config: dict, duration_ms: float, rng, with_inhibition: bool):
    """One illustrative trial at the Fig. 1E example point. The E-only and E+I panels
    differ only in whether I inputs exist and which input rate applies (the E-only rate is
    calibrated, see calibrate_rate.py), so both conditions share this one path.
    """
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
                         "`python -m block1.calibrate_rate e_plus_i` before plotting Fig. 1F.")
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
    """The bottom row shared by 1C and 1F: both cells' V_m, scale-barred."""
    ax.plot(result.t_ms, result.v_a_mV, color="black")
    ax.plot(result.t_ms, result.v_b_mV, color="0.6")
    _scale_bar(ax, float(np.ptp(np.concatenate([result.v_a_mV, result.v_b_mV]))), "mV")
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
    _scale_bar(axes[1], float(np.ptp(np.concatenate([result.i_syn_a_mV, result.i_syn_b_mV]))), "mV")
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
    _scale_bar(axes[1], float(np.ptp(np.concatenate([i_e_a, i_i_a]))), "mV")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8)
    axes[1].set_title("synaptic current (E, I separately)")

    _membrane_panel(axes[2], result)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    from config import load_config

    config = load_config("config.yaml")
    # A few hundred ms of one trial -- these panels show shape, not a measurement.
    duration_ms = float(sys.argv[1]) if len(sys.argv) > 1 else 500.0
    rng = np.random.default_rng(config["seed"])
    plot_fig1c(config, duration_ms, rng, "artifacts/fig1c.png")
    plot_fig1f(config, duration_ms, rng, "artifacts/fig1f.png")
    print("wrote fig1c/fig1f.png to artifacts/")
