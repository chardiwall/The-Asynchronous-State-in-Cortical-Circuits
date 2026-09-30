"""Block 1's Fig. 1B and 1E sweep curves, read from block1.full_pass's CSV -- one combined
panel each, c as a dashed line and r_out as a marker-line ('-o-'), matching the paper's own
Fig. 1B/E layout (dashed c / open-circle r_out on one axis, not two separate panels).
Palette: dataviz skill's validated categorical slots 1 (blue) and 2 (orange).

Fig. 1C and 1F's illustrative traces live in plot_fig1_traces.py.
"""
import csv

import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"



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


if __name__ == "__main__":
    rows = load_full_pass_csv("artifacts/block1_full_pass.csv")
    plot_fig1b(rows["fig1b"], "artifacts/fig1b.png")
    plot_fig1e(rows["fig1e_e_only"], rows["fig1e_e_plus_i"], "artifacts/fig1e.png")
    print("wrote fig1b/fig1e.png to artifacts/")
