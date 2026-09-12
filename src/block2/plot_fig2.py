"""Fig. 2 plots. Color code matches the paper's own Fig. 2B legend (E/X/Total/I)
and its reuse in 2C/2D/2E ("Color code as in (B)/(C)"): E green, I red, X blue,
Total/total-current black. 2C/2G need block2.full_pass[_current]'s aggregated
CSVs (not yet generated pending the researcher's go-ahead on the current-
correlation job); 2B/2D/2E/2G read block2.illustrative_panels' per-panel CSVs.
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

GREEN, RED, BLUE, BLACK = "#2ca02c", "#d62728", "#1f77b4", "#000000"
OUT_DIR = "artifacts/block2_illustrative"


def _read_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def plot_fig2b(csv_path: str = f"{OUT_DIR}/panel_b.csv", out_path: str = f"{OUT_DIR}/fig2b.png") -> None:
    rows = _read_csv(csv_path)
    t = [r["t_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, [r["E"] for r in rows], color=GREEN, label="E", linewidth=1)
    ax.plot(t, [r["X"] for r in rows], color=BLUE, label="X", linewidth=1)
    ax.plot(t, [r["Total"] for r in rows], color=BLACK, label="Total", linewidth=1.2)
    ax.plot(t, [r["I"] for r in rows], color=RED, label="I", linewidth=1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input currents")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2d(csv_path: str = f"{OUT_DIR}/panel_d.csv", out_path: str = f"{OUT_DIR}/fig2d.png") -> None:
    rows = _read_csv(csv_path)
    sizes = sorted({int(r["n"]) for r in rows})
    fig, axes = plt.subplots(len(sizes), 1, figsize=(6, 2.5 * len(sizes)), sharex=True)
    for ax, n in zip(axes, sizes):
        subset = [r for r in rows if int(r["n"]) == n]
        t = np.array([r["t_ms"] for r in subset])
        for pop, color in (("E", GREEN), ("I", RED), ("X", BLUE)):
            values = np.array([r[pop] for r in subset])
            z = (values - values.mean()) / (values.std() + 1e-12)
            ax.plot(t, z, color=color, label=pop, linewidth=1)
        ax.set_ylabel(f"N={n}\nz-score")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2e(csv_path: str = f"{OUT_DIR}/panel_e.csv", out_path: str = f"{OUT_DIR}/fig2e.png") -> None:
    rows = _read_csv(csv_path)
    lag = [r["lag_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lag, [r["ccg_EE"] for r in rows], color=GREEN, label="EE", linestyle="--")
    ax.plot(lag, [r["ccg_II"] for r in rows], color=RED, label="II")
    ax.plot(lag, [r["ccg_EI"] for r in rows], color="darkorange", label="EI")
    ax.plot(lag, [r["ccg_total"] for r in rows], color=BLACK, label="Total", linewidth=1.5)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Current correlation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2g(csv_path: str = f"{OUT_DIR}/panel_g.csv", out_path: str = f"{OUT_DIR}/fig2g.png") -> None:
    r_values = [row["r"] for row in _read_csv(csv_path)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r_values, bins=100, color=GREEN, alpha=0.8)
    ax.axvline(float(np.mean(r_values)), color=BLACK, linestyle="--", linewidth=1)
    ax.set_xlabel("Firing correlation r (EE pairs)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_fig2b()
    plot_fig2d()
    plot_fig2e()
    plot_fig2g()
    print(f"wrote fig2b/d/e/g.png to {OUT_DIR}/")
