"""The LSM extension's figures. Variant colours are categorical identities (validated for
CVD separation), distinct from the paper's E/I/X code, which is kept for E/I/X quantities."""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import output_path
from lib.plotting import BLACK, BLUE, GREEN, RED

VARIANT_COLOUR = {"faithful": "#7b3294", "dense_weak": "#e08214", "maass": "#018571", "input": "#8c8c8c"}
VARIANT_LABEL = {"faithful": "faithful (dense, √K-scaled)", "dense_weak": "dense, weak",
                 "maass": "Maass column", "input": "input only (no reservoir)"}


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _stats(rows, variant, key):
    """(sizes, mean, sd) of `key` over network realisations of one variant."""
    sizes = sorted({int(r["n"]) for r in rows if r["variant"] == variant})
    values = [[float(r[key]) for r in rows if r["variant"] == variant and int(r["n"]) == n] for n in sizes]
    return np.array(sizes), np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def plot_scaling(rows, out_path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    n, m, s = _stats(rows, "faithful", "r_bar")
    a.errorbar(n, m, s, fmt="o-", color=VARIANT_COLOUR["faithful"], lw=2, ms=8, capsize=3,
               label=VARIANT_LABEL["faithful"])
    a.plot(n, m[-1] * n[-1] / n, "--", color=BLACK, lw=1, label="∝ 1/N (anchored at N=5000)")
    for v in ("dense_weak", "maass"):
        nv, mv, sv = _stats(rows, v, "r_bar")
        a.errorbar(nv, mv, sv, fmt="s", color=VARIANT_COLOUR[v], ms=9, capsize=3, label=VARIANT_LABEL[v])
    a.set(xscale="log", yscale="log", xlabel="N (recurrent neurons)", ylabel="r̄ (E pairs, T = 50 ms)",
          title="Population-averaged correlation")
    a.legend(frameon=False, fontsize=8)
    _style(a)
    for key, style, label in (("sigma_r", "o-", "σ_r measured"), ("sigma_r_jittered", "o:", "σ_r jittered null")):
        n, m, s = _stats(rows, "faithful", key)
        b.errorbar(n, m, s, fmt=style, color=VARIANT_COLOUR["faithful"], lw=2, ms=7, capsize=3,
                   alpha=1.0 if key == "sigma_r" else 0.55, label=f"faithful: {label}")
    for v in ("dense_weak", "maass"):
        nv, mv, _ = _stats(rows, v, "sigma_r")
        b.plot(nv, mv, "s", color=VARIANT_COLOUR[v], ms=9, label=f"{VARIANT_LABEL[v]}: σ_r measured")
    b.set(xscale="log", yscale="log", xlabel="N (recurrent neurons)", ylabel="σ_r", title="Width of the r distribution")
    b.legend(frameon=False, fontsize=8)
    _style(b)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


CURRENT_TERMS = [("c_EE", "c_EE", GREEN), ("c_II", "c_II", RED), ("c_XX", "c_XX", BLUE),
                 ("c_EI", "2c_EI", "#555555"), ("c_EX", "2c_EX", "#999999"), ("c_IX", "2c_IX", "#bbbbbb"),
                 ("c_total", "c (total)", BLACK)]


def plot_currents(rows, size, out_path):
    variants = [v for v in ("faithful", "dense_weak", "maass") if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.8), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), variants):
        sel = [r for r in rows if r["variant"] == v and int(r["n"]) == size]
        values = [np.mean([float(r[k]) for r in sel]) * (2 if label.startswith("2") else 1)
                  for k, label, _ in CURRENT_TERMS]
        ax.bar(range(len(values)), values, color=[c for *_, c in CURRENT_TERMS], width=0.75, edgecolor="white")
        for i, val in enumerate(values):
            ax.text(i, val, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
        ax.axhline(0, color=BLACK, lw=0.8)
        ax.set_xticks(range(len(values)), [label for _, label, _ in CURRENT_TERMS], rotation=45, fontsize=8)
        ax.set_title(f"{VARIANT_LABEL[v]}, N={size}", fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("current-correlation term")
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_current_scaling(rows, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    for key, label, colour in CURRENT_TERMS:
        n, m, s = _stats(rows, "faithful", key)
        factor = 2 if label.startswith("2") else 1
        ax.errorbar(n, factor * m, factor * s, fmt="o-", color=colour, lw=2 if key == "c_total" else 1.2,
                    ms=6, capsize=2, label=label)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set(xscale="log", xlabel="N (recurrent neurons)", ylabel="current-correlation term",
           title="Faithful LSM: cancellation vs N")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_readout(rows, out_path):
    datasets = sorted({r["dataset"] for r in rows})
    order = [v for v in ("input", "faithful", "dense_weak", "maass") if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3.8))
    for ax, d in zip(np.atleast_1d(axes), datasets):
        acc = [next((100 * float(r["test_accuracy"]) for r in rows if r["dataset"] == d and r["variant"] == v), np.nan)
               for v in order]
        ax.bar(range(len(order)), acc, color=[VARIANT_COLOUR[v] for v in order], width=0.7, edgecolor="white")
        for i, a in enumerate(acc):
            ax.text(i, a + 0.5, f"{a:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(order)), [VARIANT_LABEL[v].replace(" (", "\n(") for v in order], fontsize=7)
        ax.set(ylabel="test accuracy (%)", ylim=(0, 105), title=d)
        _style(ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def main(config: dict) -> None:
    out_dir = output_path(config, "lsm_figures_dir")
    os.makedirs(out_dir, exist_ok=True)
    asynchrony_csv = output_path(config, "lsm_asynchrony_csv")
    if os.path.exists(asynchrony_csv):
        rows = _rows(asynchrony_csv)
        plot_scaling(rows, f"{out_dir}/lsm_scaling.png")
        plot_currents(rows, config["lsm"]["size"], f"{out_dir}/lsm_currents.png")
        plot_current_scaling(rows, f"{out_dir}/lsm_current_scaling.png")
    readout_csv = output_path(config, "lsm_readout_csv")
    if os.path.exists(readout_csv):
        plot_readout(_rows(readout_csv), f"{out_dir}/lsm_readout.png")
    print(f"LSM figures in {out_dir}/")
