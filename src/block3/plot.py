"""Fig. 3's four panels.

3A/3B read block3.figures_ab' CSVs; 3C/3D read block3.figures_cd --aggregate's CSV. Colours are
the paper's own Fig. 3 legend: E green, I red, X blue; 3C's EPSP condition green, IPSP
red, the mixed EPSP-IPSP condition gold, and rest black.

Usage: python -m block3.plot
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import load_config, output_path
from lib.plotting import BLACK, BLUE, GOLD, GREEN, RED

def _read_csv(path: str) -> list[dict]:
    """Rows as dicts with values left as STRINGS -- deliberately not lib.plotting.read_csv,
    which floats every value. Fig. 3B's CSV carries a `kind` column of "measured"/"jittered",
    so float-casting every column would raise. Callers float the numeric columns they use.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def plot_fig3a(raster_csv: str, tracking_csv: str, out_path: str) -> None:
    """Raster of the rate-sorted neuron sample (top) over the z-scored population
    activities that show E, I and X tracking one another (bottom).
    """
    raster = _read_csv(raster_csv)
    tracking = _read_csv(tracking_csv)

    by_row = defaultdict(list)
    population_of = {}
    for spike in raster:
        row = int(spike["row"])
        by_row[row].append(float(spike["t_ms"]))
        population_of[row] = spike["population"]
    rows = sorted(by_row)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[3, 1], sharex=True)
    axes[0].eventplot([by_row[r] for r in rows],
                       colors=[GREEN if population_of[r] == "E" else RED for r in rows],
                       linelengths=0.9, linewidths=0.4)
    axes[0].set_ylabel("Neuron (sorted by rate)")
    axes[0].set_title("Fig. 3A")

    t = [float(r["t_ms"]) for r in tracking]
    for population, color in (("E", GREEN), ("I", RED), ("X", BLUE)):
        axes[1].plot(t, [float(r[population]) for r in tracking], color=color,
                     label=population, linewidth=0.8)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("Activity (z-score)")
    axes[1].legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig3b(correlations_csv: str, out_path: str) -> None:
    """Measured pairwise spike-count correlations (black) against their jittered-surrogate
    null (grey), on one shared set of bins so the widths are directly comparable.
    """
    rows = _read_csv(correlations_csv)
    measured = np.array([float(r["r"]) for r in rows if r["kind"] == "measured"])
    jittered = np.array([float(r["r"]) for r in rows if r["kind"] == "jittered"])
    measured, jittered = measured[np.isfinite(measured)], jittered[np.isfinite(jittered)]

    bins = np.linspace(min(measured.min(), jittered.min()),
                       max(measured.max(), jittered.max()), 120)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(jittered, bins=bins, color="0.7", label="jittered surrogates")
    ax.hist(measured, bins=bins, histtype="step", color=BLACK, label="measured")
    ax.axvline(float(np.mean(measured)), color=BLACK, linestyle="--", linewidth=1)
    ax.set_xlabel("Spike count correlation r (EE pairs, T = 50 ms)")
    ax.set_ylabel("Number of pairs")
    ax.set_title(rf"Fig. 3B ($\bar{{r}}$ = {np.mean(measured):.5f}, $\sigma_r$ = {np.std(measured):.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _condition_curves(rows: list[dict], config: dict) -> list[tuple[str, str, list[dict]]]:
    """(label, colour, rows) for Fig. 3C's four curves, selected from the swept tasks by
    the current levels config names as the EPSP, IPSP and rest conditions.
    """
    fig3cd = config["spiking_network"]["fig3cd"]
    epsp, ipsp, rest = (fig3cd["epsp_condition_nA"], fig3cd["ipsp_condition_nA"],
                        fig3cd["rest_condition_nA"])

    def select(a, b, group):
        return [r for r in rows if float(r["i_app_a_nA"]) == a
                and float(r["i_app_b_nA"]) == b and r["group"] == group]

    return [
        ("EPSP-EPSP", GREEN, select(epsp, epsp, "within_a")),
        ("IPSP-IPSP", RED, select(ipsp, ipsp, "within_a")),
        ("EPSP-IPSP", GOLD, select(epsp, ipsp, "cross")),
        ("Rest", BLACK, select(rest, rest, "within_a")),
    ]


def _mean_over_networks(rows: list[dict], value_key: str) -> tuple[np.ndarray, np.ndarray]:
    grouped = defaultdict(list)
    for r in rows:
        grouped[float(r["lag_ms"])].append(float(r[value_key]))
    lags = np.array(sorted(grouped))
    return lags, np.array([np.nanmean(grouped[lag]) for lag in lags])


def plot_fig3c(csv_path: str, out_path: str) -> None:
    rows = _read_csv(csv_path)
    config = load_config("config.yaml")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color, subset in _condition_curves(rows, config):
        if not subset:
            continue
        lags, ccg = _mean_over_networks(subset, "ccg")
        ax.plot(lags, ccg, color=color, label=label, linewidth=1.3)
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Membrane potential CCG")
    ax.set_title("Fig. 3C")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig3d(csv_path: str, out_path: str) -> None:
    """Peak CCG height against the pair's mean holding potential.

    The paper's Fig. 3D is a V-shape that stays ENTIRELY NON-NEGATIVE: about +0.27 at the
    inhibitory reversal, a minimum near +0.02 around rest, rising to about +0.15 at the
    excitatory reversal. A negative point here means something is wrong -- most likely
    signed_peak picking a side-lobe -- not a reproduction of the paper.

    The large negative correlation belongs to the EPSP-IPSP condition, which the paper
    shows only in Fig. 3C (the gold curve), not here.
    """
    rows = [r for r in _read_csv(csv_path)
            if r["group"] == "within_a" and float(r["i_app_a_nA"]) == float(r["i_app_b_nA"])]
    config = load_config("config.yaml")
    fig3cd = config["spiking_network"]["fig3cd"]

    grouped = defaultdict(list)
    for r in rows:
        grouped[float(r["i_app_a_nA"])].append((float(r["holding_mV"]), float(r["peak"])))
    levels = sorted(grouped)
    holding = np.array([np.mean([h for h, _ in grouped[lvl]]) for lvl in levels])
    peak = np.array([np.mean([p for _, p in grouped[lvl]]) for lvl in levels])

    highlight = {fig3cd["epsp_condition_nA"]: GREEN, fig3cd["ipsp_condition_nA"]: RED,
                 fig3cd["rest_condition_nA"]: BLACK}
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(holding, peak, "-", color="0.5", linewidth=1, zorder=1)
    ax.scatter(holding, peak, c=[highlight.get(lvl, "0.5") for lvl in levels], s=45, zorder=2)
    ax.axhline(0, color="0.8", linewidth=0.8)
    ax.set_xlabel("Mean holding potential (mV)")
    ax.set_ylabel("Peak of membrane potential CCG")
    ax.set_title("Fig. 3D")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(config: dict) -> None:
    """Every Fig. 3 panel, with each path resolved from config. Called both by this
    module's __main__ and by `block3.run plot`, so the two cannot drift apart.
    """
    panels_dir = output_path(config, "block3_panels_dir")
    os.makedirs(panels_dir, exist_ok=True)
    plot_fig3a(output_path(config, "block3_fig3a_raster_csv"),
               output_path(config, "block3_fig3a_tracking_csv"),
               output_path(config, "block3_fig3a"))
    plot_fig3b(output_path(config, "block3_fig3b_correlations_csv"),
               output_path(config, "block3_fig3b"))
    plot_fig3c(output_path(config, "block3_vm_ccg_csv"), output_path(config, "block3_fig3c"))
    plot_fig3d(output_path(config, "block3_vm_ccg_csv"), output_path(config, "block3_fig3d"))
    print(f"wrote fig3a/b/c/d.png to {panels_dir}/")


if __name__ == "__main__":
    main(load_config("config.yaml"))
