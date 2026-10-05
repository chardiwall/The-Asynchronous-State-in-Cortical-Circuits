"""Fig. 2's five data panels (B, C, D, E, G). 2A and 2F are schematics in the paper, not
simulations, so nothing is plotted for them.

Color code is the paper's own Fig. 2B legend, reused by 2C/2D/2E ("Color code as in
(B)/(C)"): E green, I red, X blue, Total black, and the mixed EI quantity orange.

2B/2D/2E/2G read block2.panels' per-panel CSVs. Fig. 2C is different in kind
-- it reads the two aggregated full-pass CSVs -- so it lives in plot_sweep.py; running
this module draws all five.

Usage: python -m block2.plot
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from config import load_config
from lib.plotting import BLACK, BLUE, GREEN, ORANGE, RED, read_csv

EI_LAG_ZOOM_MS = 5.0   # half-width of 2D's and 2E's EI-Lag insets


def _by_size(rows: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["n"])].append(row)
    return dict(sorted(grouped.items()))


def plot_fig2b(csv_path: str, out_path: str) -> None:
    rows = read_csv(csv_path)
    t = [r["t_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for key, color, width in (("E", GREEN, 1), ("X", BLUE, 1), ("Total", BLACK, 1.2), ("I", RED, 1)):
        ax.plot(t, [r[key] for r in rows], color=color, label=key, linewidth=width)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, label="threshold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input currents")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2d(csv_path: str, out_path: str) -> None:
    """z-scored m_E/m_I/m_X per network size, with an inset magnifying one instance of
    the E-to-I lag (the paper's "EI-Lag") in each panel.
    """
    grouped = _by_size(read_csv(csv_path))
    fig, axes = plt.subplots(len(grouped), 1, figsize=(6, 2.8 * len(grouped)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (n, subset) in zip(axes, grouped.items()):
        t = np.array([r["t_ms"] for r in subset])
        inset = ax.inset_axes([0.62, 0.58, 0.36, 0.4])
        mid = t[len(t) // 2]
        for pop, color in (("E", GREEN), ("I", RED), ("X", BLUE)):
            values = np.array([r[pop] for r in subset])
            z = (values - values.mean()) / (values.std() + 1e-12)
            ax.plot(t, z, color=color, label=pop, linewidth=1)
            inset.plot(t, z, color=color, linewidth=1)
        inset.set_xlim(mid - EI_LAG_ZOOM_MS, mid + EI_LAG_ZOOM_MS)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("EI-Lag", fontsize=7)
        ax.set_ylabel(f"N={n}\nz-score")
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2e(csv_path: str, out_path: str,
                display_lag_ms: float | None = None) -> None:
    """Population-averaged CCGs of the current COMPONENTS at the largest N, with two insets
    across every N: the total CCG's peak (top) and the EI/IE peaks (bottom), which is where
    the EI-Lag shrinking with N is visible.

    IE is read from the data, not derived by reversing EI. The two are related by
    CCG_IE(lag) == CCG_EI(-lag), but they are genuinely different curves at non-zero lag,
    and their asymmetry about zero IS the EI-Lag the inset exists to show.
    """
    if display_lag_ms is None:
        display_lag_ms = load_config("config.yaml")["binary_network"]["ccg"]["display_lag_ms"]
    grouped = _by_size(read_csv(csv_path))
    largest = max(grouped)
    rows = grouped[largest]
    lag = np.array([r["lag_ms"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for field, colour, style in (("c_EE", GREEN, "--"), ("c_II", RED, "-"),
                                  ("c_XX", BLUE, "-"), ("c_EI", ORANGE, "-"),
                                  ("c_IE", "#7b3294", ":")):
        ax.plot(lag, [r[field] for r in rows], color=colour, label=field[2:], linestyle=style)
    ax.plot(lag, [r["c_total"] for r in rows], color=BLACK, label="Total", linewidth=1.5)
    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.set_xlim(-display_lag_ms, display_lag_ms)
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("Current correlation")
    ax.set_title(f"Fig. 2E (N={largest})")
    ax.legend(fontsize=8, loc="upper left")

    total_inset = ax.inset_axes([0.66, 0.60, 0.32, 0.34])
    ei_inset = ax.inset_axes([0.66, 0.14, 0.32, 0.34])
    shades = np.linspace(0.65, 0.0, len(grouped))
    for shade, (n, subset) in zip(shades, grouped.items()):
        n_lag = np.array([r["lag_ms"] for r in subset])
        total_inset.plot(n_lag, [r["c_total"] for r in subset], color=str(shade),
                         linewidth=1, label=f"N={n}")
        ei_inset.plot(n_lag, [r["c_EI"] for r in subset], color=ORANGE,
                      alpha=1 - shade, linewidth=1)
        ei_inset.plot(n_lag, [r["c_IE"] for r in subset], color="#7b3294",
                      alpha=1 - shade, linewidth=1)
    for inset, title in ((total_inset, "Total, peak"), (ei_inset, "EI / IE peaks")):
        inset.set_xlim(-EI_LAG_ZOOM_MS, EI_LAG_ZOOM_MS)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(title, fontsize=7)
    total_inset.legend(fontsize=6, frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2g(csv_path: str, out_path: str) -> None:
    r_values = [row["r"] for row in read_csv(csv_path)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(r_values, bins=100, color=GREEN, alpha=0.8)
    ax.axvline(float(np.mean(r_values)), color=BLACK, linestyle="--", linewidth=1)
    ax.set_xlabel("Firing correlation r (EE pairs)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from block2.plot_sweep import plot_fig2c
    from config import load_config, output_path

    config = load_config("config.yaml")
    panels_dir = output_path(config, "block2_panels_dir")
    plot_fig2b(f"{panels_dir}/panel_b.csv", output_path(config, "block2_fig2b"))
    plot_fig2d(f"{panels_dir}/panel_d.csv", output_path(config, "block2_fig2d"))
    plot_fig2e(f"{panels_dir}/panel_e.csv", output_path(config, "block2_fig2e"))
    plot_fig2g(f"{panels_dir}/panel_g.csv", output_path(config, "block2_fig2g"))
    plot_fig2c(output_path(config, "block2_sweep_csv"),
               output_path(config, "block2_current_csv"),
               output_path(config, "block2_fig2c"))
    print(f"wrote fig2b/c/d/e/g.png to {panels_dir}/")
