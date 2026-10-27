"""The LSM extension's figures. Variant colours are categorical identities (validated for
CVD separation), distinct from the paper's E/I/X code, which is kept for E/I/X quantities."""
import csv
import os

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import output_path
from lib.plotting import BLACK, BLUE, GREEN, RED

VARIANT_COLOUR = {"faithful": "#7b3294", "dense_weak": "#e08214", "maass": "#018571", "input": "#8c8c8c",
                  "maass2002": "#2166ac", "kaiser2017": "#c51b7d", "wijesinghe2019": "#8c510a"}
DATASET_LABEL = {"nmnist": "N-MNIST", "dvs_gesture": "DVS Gesture"}
VARIANT_LABEL = {"faithful": "faithful (dense, √K-scaled)", "dense_weak": "dense, weak",
                 "maass": "Maass column (Renart neuron)", "input": "input only (no reservoir)",
                 "maass2002": "Maass 2002", "kaiser2017": "Kaiser 2017", "wijesinghe2019": "Wijesinghe 2019"}


def _ticks(ax, order):
    """Variant names under bars; angled once there are more bars than fit flat."""
    ax.set_xticks(range(len(order)), [VARIANT_LABEL[v].replace(" (", "\n(") for v in order], fontsize=7,
                  rotation=30 if len(order) > 4 else 0, ha="right" if len(order) > 4 else "center")


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _stats(rows, variant, key):
    """(sizes, mean, sd) of `key` over network realisations of one variant."""
    sizes = sorted({int(r["n"]) for r in rows if r["variant"] == variant})
    values = [[float(r[key]) for r in rows if r["variant"] == variant and int(r["n"]) == n] for n in sizes]
    return np.array(sizes), np.array([np.mean(v) for v in values]), np.array([np.std(v) for v in values])


def _style(ax, sizes=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)
    if sizes is not None:  # label only the simulated N, not the log axis' minor ticks
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


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
    _style(a, n)
    for key, style, label in (("sigma_r", "o-", "σ_r measured"), ("sigma_r_jittered", "o:", "σ_r jittered null")):
        n, m, s = _stats(rows, "faithful", key)
        b.errorbar(n, m, s, fmt=style, color=VARIANT_COLOUR["faithful"], lw=2, ms=7, capsize=3,
                   alpha=1.0 if key == "sigma_r" else 0.55, label=f"faithful: {label}")
    for v in ("dense_weak", "maass"):
        nv, mv, _ = _stats(rows, v, "sigma_r")
        b.plot(nv, mv, "s", color=VARIANT_COLOUR[v], ms=9, label=f"{VARIANT_LABEL[v]}: σ_r measured")
    b.set(xscale="log", yscale="log", xlabel="N (recurrent neurons)", ylabel="σ_r", title="Width of the r distribution")
    b.legend(frameon=False, fontsize=8)
    b.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    _style(b, n)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


CURRENT_TERMS = [("c_EE", "c_EE", GREEN), ("c_II", "c_II", RED), ("c_XX", "c_XX", BLUE),
                 ("c_EI", "2c_EI", "#555555"), ("c_EX", "2c_EX", "#999999"), ("c_IX", "2c_IX", "#bbbbbb"),
                 ("c_total", "c (total)", BLACK)]


def plot_currents(rows, size, out_path, suffix="", order=("faithful", "dense_weak", "maass")):
    variants = [v for v in order if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(variants), figsize=(4 * len(variants), 3.8), sharey=True)
    for ax, v in zip(np.atleast_1d(axes), variants):
        sel = [r for r in rows if r["variant"] == v and int(r["n"]) == size]
        values = [np.mean([float(r[k + suffix]) for r in sel]) * (2 if label.startswith("2") else 1)
                  for k, label, _ in CURRENT_TERMS]
        ax.bar(range(len(values)), values, color=[c for *_, c in CURRENT_TERMS], width=0.75, edgecolor="white")
        for i, val in enumerate(values):
            ax.text(i, val, f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=7)
        ax.axhline(0, color=BLACK, lw=0.8)
        ax.set_xticks(range(len(values)), [label for _, label, _ in CURRENT_TERMS], rotation=45, fontsize=8)
        ax.set_title(f"{VARIANT_LABEL[v]}, N={size}", fontsize=9)
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("current-correlation term " + {"": "(1 ms)", "_T": "(T-averaged)",
                                      "_noise": "(trial-to-trial noise, T-averaged)"}[suffix])
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_current_scaling(rows, out_path, suffix=""):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for key, label, colour in CURRENT_TERMS:
        n, m, s = _stats(rows, "faithful", key + suffix)
        factor = 2 if label.startswith("2") else 1
        ax.errorbar(n, factor * m, factor * s, fmt="o-", color=colour, lw=2 if key == "c_total" else 1.2,
                    ms=6, capsize=2, label=label)
    ax.axhline(0, color=BLACK, lw=0.6)
    ax.set(xscale="log", xlabel="N (recurrent neurons)", ylabel="current-correlation term",
           title="Faithful LSM: cancellation vs N\n" + ("(currents averaged over T = 50 ms)" if suffix else "(1 ms currents)"))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _style(ax, n)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_readout(rows, out_path, order=("input", "faithful", "dense_weak", "maass")):
    datasets = sorted({r["dataset"] for r in rows})
    order = [v for v in order if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(4.5, 0.9 * len(order)) * len(datasets), 3.8))
    for ax, d in zip(np.atleast_1d(axes), datasets):
        found = [next((r for r in rows if r["dataset"] == d and r["variant"] == v), None) for v in order]
        acc = np.array([100 * float(r["test_accuracy"]) if r else np.nan for r in found])
        # 95% binomial interval of a test accuracy over n_test independent samples
        ci = np.array([196 * np.sqrt(float(r["test_accuracy"]) * (1 - float(r["test_accuracy"])) / float(r["n_test"]))
                       if r else np.nan for r in found])
        ax.bar(range(len(order)), acc, yerr=ci, capsize=4, color=[VARIANT_COLOUR[v] for v in order],
               width=0.7, edgecolor="white", error_kw={"lw": 1})
        for i, (a, e) in enumerate(zip(acc, ci)):
            ax.text(i, a + e + 0.5, f"{a:.1f}%", ha="center", va="bottom", fontsize=9)
        _ticks(ax, order)
        n_test = int(float(found[0]["n_test"])) if found[0] else 0
        ax.set(ylabel="test accuracy (%)", ylim=(0, 108), title=f"{DATASET_LABEL.get(d, d)} (n_test = {n_test}, ±95% CI)")
        _style(ax)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_size_effect(rows, base_size, out_path):
    """Test accuracy of faithful vs maass at lsm.size and at every <variant>_N<N> size run."""
    datasets = sorted({r["dataset"] for r in rows})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3.6))
    for ax, d in zip(np.atleast_1d(axes), datasets):
        for v in ("faithful", "maass"):
            points = sorted((base_size if r["variant"] == v else int(r["variant"].split("_N")[1]),
                             100 * float(r["test_accuracy"]), int(float(r["n_test"])))
                            for r in rows if r["dataset"] == d and r["variant"].split("_N")[0] == v)
            n, acc, n_test = (np.array(x) for x in zip(*points))
            ci = 196 * np.sqrt(acc / 100 * (1 - acc / 100) / n_test)
            ax.errorbar(n, acc, ci, fmt="o-", color=VARIANT_COLOUR[v], lw=2, ms=8, capsize=4, label=VARIANT_LABEL[v])
            for x, a in zip(n, acc):
                ax.annotate(f"{a:.1f}%", (x, a), textcoords="offset points", xytext=(8, -4), fontsize=8)
        ax.set(xscale="log", xlabel="N (recurrent neurons)", ylabel="test accuracy (%)", title=DATASET_LABEL.get(d, d))
        ax.legend(frameon=False, fontsize=8)
        _style(ax, n)
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def plot_reliability(rows, out_path, order=("input", "faithful", "dense_weak", "maass")):
    """Stimulus-locked share of each liquid's state (lsm.readout.reliability), per dataset."""
    datasets = sorted({r["dataset"] for r in rows})
    order = [v for v in order if any(r["variant"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(4.5, 0.9 * len(order)) * len(datasets), 3.6), sharey=True)
    for ax, d in zip(np.atleast_1d(axes), datasets):
        value = [next((float(r["reliability"]) for r in rows if r["dataset"] == d and r["variant"] == v), np.nan)
                 for v in order]
        ax.bar(range(len(order)), value, color=[VARIANT_COLOUR[v] for v in order], width=0.7, edgecolor="white")
        for i, x in enumerate(value):
            ax.text(i, x + 0.02, f"{x:.2f}", ha="center", va="bottom", fontsize=9)
        _ticks(ax, order)
        ax.set(ylim=(0, 1.1), title=DATASET_LABEL.get(d, d))
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("stimulus-locked share of state variance")
    fig.tight_layout(); fig.savefig(out_path, dpi=180); plt.close(fig)


def main(config: dict) -> None:
    out_dir = output_path(config, "lsm_figures_dir")
    os.makedirs(out_dir, exist_ok=True)
    asynchrony_csv = output_path(config, "lsm_asynchrony_csv")
    if os.path.exists(asynchrony_csv):
        rows = _rows(asynchrony_csv)
        plot_scaling(rows, f"{out_dir}/lsm_scaling.png")
        for suffix, tag in (("_T", "_T"), ("", "_1ms")):
            if f"c_total{suffix}" in rows[0] and rows[0][f"c_total{suffix}"]:
                plot_currents(rows, config["lsm"]["size"], f"{out_dir}/lsm_currents{tag}.png", suffix)
                plot_current_scaling(rows, f"{out_dir}/lsm_current_scaling{tag}.png", suffix)
    import glob
    import json

    reliability = [json.load(open(p)) for p in
                   sorted(glob.glob(f"{output_path(config, 'lsm_features_dir')}/*/reliability_*.json"))]
    if reliability:
        plot_reliability([r for r in reliability if "_N" not in r["variant"]], f"{out_dir}/lsm_reliability.png")
    readout_csv = output_path(config, "lsm_readout_csv")
    if os.path.exists(readout_csv):
        readout = _rows(readout_csv)
        plot_readout([r for r in readout if "_N" not in r["variant"]], f"{out_dir}/lsm_readout.png")
        if any("_N" in r["variant"] for r in readout):
            plot_size_effect(readout, config["lsm"]["size"], f"{out_dir}/lsm_size_effect.png")
    print(f"LSM figures in {out_dir}/")
