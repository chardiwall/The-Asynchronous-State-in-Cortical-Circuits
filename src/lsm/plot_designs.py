"""Figures putting the three published liquids (lsm.designs) beside the Renart-derived
variants, and every liquid's correlation structure beside its accuracy.

  lsm_designs_currents_T.png        background c-decomposition (T-averaged), per liquid
  lsm_designs_driven_<dataset>.png  driven-state NOISE c-decomposition, per liquid
  lsm_designs_readout.png           test accuracy, every liquid, both datasets
  lsm_designs_reliability.png       stimulus-locked share of the liquid state
  lsm_designs_vs_accuracy.png       accuracy against each correlation measure, one point per
                                    liquid (Spearman rho over 6 points: a trend, not a test)

    python -m lsm.plot_designs
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from config import load_config, output_path
from lsm.plot import (DATASET_LABEL, VARIANT_COLOUR, VARIANT_LABEL, _rows, _style, plot_currents,
                      plot_readout, plot_reliability)

LIQUIDS = ("faithful", "dense_weak", "maass", "maass2002", "kaiser2017", "wijesinghe2019")
MEASURES = [("background", "r_bar", "r̄, background"),
            ("background", "c_II_T", "c_II, background"),
            ("background", "c_EI_T", "c_EI, background"),
            ("background", "c_total_T", "c (total), background"),
            ("driven", "r_noise", "r noise, driven"),
            ("driven", "c_total_noise", "c noise (total), driven")]


def _background(rows, size):
    """{variant: {key: mean over networks}} at N = size."""
    out = {}
    for v in LIQUIDS:
        sel = [r for r in rows if r["variant"] == v and int(r["n"]) == size]
        if sel:
            out[v] = {k: float(np.nanmean([float(r[k]) for r in sel])) for k in sel[0]
                      if k.startswith(("r_bar", "c_", "rate_"))}
    return out


def plot_vs_accuracy(background, driven, readout, out_path):
    datasets = sorted({r["dataset"] for r in readout})
    fig, axes = plt.subplots(len(datasets), len(MEASURES), figsize=(2.6 * len(MEASURES), 2.8 * len(datasets)),
                             squeeze=False)
    for row_axes, d in zip(axes, datasets):
        accuracy = {r["variant"]: 100 * float(r["test_accuracy"]) for r in readout if r["dataset"] == d}
        for ax, (kind, key, label) in zip(row_axes, MEASURES):
            source = background if kind == "background" else driven.get(d, {})
            points = [(v, source[v][key], accuracy[v]) for v in LIQUIDS
                      if v in accuracy and v in source and np.isfinite(source[v].get(key, np.nan))]
            for v, x, y in points:
                ax.plot(x, y, "o", ms=8, color=VARIANT_COLOUR[v], markeredgecolor="white")
            if len(points) > 2:
                rho = spearmanr([p[1] for p in points], [p[2] for p in points])[0]
                ax.set_title(f"{label}\nSpearman ρ = {rho:+.2f} (n={len(points)})", fontsize=8)
            else:
                ax.set_title(label, fontsize=8)
            ax.set_xlabel(key, fontsize=8)
            _style(ax)
        row_axes[0].set_ylabel(f"{DATASET_LABEL.get(d, d)}\ntest accuracy (%)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=VARIANT_COLOUR[v], label=VARIANT_LABEL[v]) for v in LIQUIDS]
    fig.legend(handles=handles, loc="lower center", ncol=len(LIQUIDS), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(out_path, dpi=180); plt.close(fig)


def main(config: dict) -> None:
    out_dir = output_path(config, "lsm_figures_dir")
    features = output_path(config, "lsm_features_dir")
    size = config["lsm"]["size"]
    os.makedirs(out_dir, exist_ok=True)
    asynchrony = _rows(output_path(config, "lsm_asynchrony_csv"))
    plot_currents(asynchrony, size, f"{out_dir}/lsm_designs_currents_T.png", "_T", order=LIQUIDS)
    driven = {}
    for path in sorted(glob.glob(f"{features}/*/driven_*.json")):
        row = json.load(open(path))
        driven.setdefault(row["dataset"], {})[row["variant"]] = row
    for d, rows in driven.items():
        plot_currents([{**r, "n": size} for r in rows.values()], size, f"{out_dir}/lsm_designs_driven_{d}.png",
                      "_noise", order=LIQUIDS)
    readout = [r for r in _rows(output_path(config, "lsm_readout_csv")) if "_N" not in r["variant"]]
    plot_readout(readout, f"{out_dir}/lsm_designs_readout.png", order=("input", *LIQUIDS))
    reliability = [json.load(open(p)) for p in sorted(glob.glob(f"{features}/*/reliability_*.json"))]
    plot_reliability([r for r in reliability if "_N" not in r["variant"]], f"{out_dir}/lsm_designs_reliability.png",
                     order=("input", *LIQUIDS))
    plot_vs_accuracy(_background(asynchrony, size), driven, readout, f"{out_dir}/lsm_designs_vs_accuracy.png")
    print(f"design figures in {out_dir}/")


if __name__ == "__main__":
    main(load_config("config.yaml"))
