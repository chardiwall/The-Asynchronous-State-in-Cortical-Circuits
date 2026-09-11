"""Plots Fig. 1B and Fig. 1E from the paper-accurate full-pass results in
artifacts/metrics.jsonl (L=10,000s). Palette: dataviz skill's validated categorical
slots 1 (blue, #2a78d6) and 2 (orange, #eb6834) -- confirmed CVD-safe as an adjacent
pair. One axis per chart (c and r_out plotted separately, never dual-axis).
"""
import json
import sys

sys.path.insert(0, "src")

import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"


def load_full_pass():
    records = {"fig1b": [], "fig1e_e_only": [], "fig1e_e_plus_i": []}
    with open("artifacts/metrics.jsonl") as f:
        for line in f:
            r = json.loads(line)
            phase = r.get("phase", "")
            for key in records:
                if phase == f"block1_full_pass_{key}":
                    records[key].append(r)
    for key in records:
        records[key].sort(key=lambda r: r.get("p", r.get("r_in")))
    return records


def plot_fig1b(records, out_path):
    xs = [r["p"] for r in records]
    c = [r["c"] for r in records]
    r_out = [r["r_out"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, ys, label in [(axes[0], c, "c"), (axes[1], r_out, "r_out")]:
        ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=8)
        ax.set_xlabel("p (shared fraction)")
        ax.set_ylabel(label)
        ax.axhline(0, color="0.8", linewidth=1, zorder=0)
    axes[0].set_title("Fig. 1B: current correlation c")
    axes[1].set_title("Fig. 1B: spike correlation r_out")
    fig.suptitle("Fig. 1B (E-only, r_in=0, L=10,000s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_fig1e(e_only, e_plus_i, out_path):
    r_in_only = [r["r_in"] for r in e_only]
    c_only = [r["c"] for r in e_only]
    rout_only = [r["r_out"] for r in e_only]
    r_in_ei = [r["r_in"] for r in e_plus_i]
    c_ei = [r["c"] for r in e_plus_i]
    rout_ei = [r["r_out"] for r in e_plus_i]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, y_only, y_ei, label in [
        (axes[0], c_only, c_ei, "c"), (axes[1], rout_only, rout_ei, "r_out"),
    ]:
        ax.plot(r_in_only, y_only, color=BLUE, linewidth=2, marker="o", markersize=8, label="E-only")
        ax.plot(r_in_ei, y_ei, color=ORANGE, linewidth=2, marker="o", markersize=8, label="E+I")
        ax.set_xlabel("r_in")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    axes[0].set_title("Fig. 1E: current correlation c")
    axes[1].set_title("Fig. 1E: spike correlation r_out")
    fig.suptitle("Fig. 1E (p=0.2, L=10,000s): E-only vs E+I")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    records = load_full_pass()
    plot_fig1b(records["fig1b"], "artifacts/fig1b_full_pass.png")
    plot_fig1e(records["fig1e_e_only"], records["fig1e_e_plus_i"], "artifacts/fig1e_full_pass.png")
    print("saved artifacts/fig1b_full_pass.png and artifacts/fig1e_full_pass.png")
