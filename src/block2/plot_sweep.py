"""Fig. 2C: the log-log N-sweep of every correlation this block measures. Reads the two
aggregated CSVs (block2.run --aggregate and block2.run --aggregate),
unlike the other four panels, which read block2.panels' short per-panel runs.

This is the block's central result: r_bar falling along 1/N while c falls along 1/sqrt(N)
and the components stay O(1).
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from block2.plot import BLACK, BLUE, GREEN, ORANGE, OUT_DIR, RED, _read_csv


def _label(name: str) -> str:
    return rf"$c_{{{name[2:]}}}$"


def _mean_by_size(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    """(sizes, mean of `key` over that size's realisations) -- how each dot in 2C is built."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["n"])].append(row[key])
    sizes = np.array(sorted(grouped))
    return sizes, np.array([np.mean(grouped[n]) for n in sizes])


def _plot_signed(ax, sizes, values, color, label, hollow: bool = False) -> None:
    """Plots |values| on the log axis, but draws any NEGATIVE point as a hollow marker with
    a heavier edge so a sign flip is visible rather than silently mirrored into signal.
    """
    values = np.asarray(values)
    negative = values < 0
    ax.loglog(sizes, np.abs(values), "s", color=color, label=label,
              markerfacecolor="none" if hollow else color)
    if negative.any():
        ax.loglog(sizes[negative], np.abs(values[negative]), "s", color=color,
                  markerfacecolor="white", markeredgewidth=1.6, linestyle="none")


def plot_fig2c(
    firing_csv: str = "artifacts/block2_full_pass.csv",
    current_csv: str = "artifacts/block2_full_pass_current.csv",
    out_path: str = f"{OUT_DIR}/fig2c.png",
) -> None:
    """r (hollow squares), the total current correlation c (filled squares) and the six
    current components versus N, log-log, with the paper's 1/N and 1/sqrt(N) guide lines.

    Components of one cell's current, not currents of different populations of cells --
    see block2.measure.current_component_correlations. c_EI, c_EX and c_IX are negative (they
    are what cancels the positive c_EE + c_II + c_XX), and a log axis cannot show a
    negative value, so every series is drawn at |value| with negative points marked as
    hollow rather than silently mirrored into apparent signal.

    A non-finite value raises instead of vanishing: one silent neuron NaNs a whole
    realisation, and a NaN at the anchor size would otherwise delete an entire guide line
    without saying so.
    """
    firing = _read_csv(firing_csv)
    current = _read_csv(current_csv)

    # The two CSVs come from two independent Slurm arrays and each aggregates whatever task
    # output exists, so partial completion is normal and their size sets can differ. Plot
    # only sizes present in both: otherwise the components line up against the wrong N and
    # the guide lines anchor at a different network size, with no error.
    firing_sizes = {int(row["n"]) for row in firing}
    current_sizes = {int(row["n"]) for row in current}
    shared = sorted(firing_sizes & current_sizes)
    if not shared:
        raise ValueError(
            f"the two sweeps share no network size: firing has {sorted(firing_sizes)}, "
            f"current has {sorted(current_sizes)}. Aggregate both before plotting."
        )
    if firing_sizes != current_sizes:
        print(f"Fig. 2C: plotting the {len(shared)} sizes present in both sweeps; "
              f"firing-only {sorted(firing_sizes - current_sizes)}, "
              f"current-only {sorted(current_sizes - firing_sizes)}", flush=True)

    missing = {"c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total"} - set(current[0])
    if missing:
        raise ValueError(
            f"{current_csv} lacks {sorted(missing)}. It predates the 2026-09-30 correction "
            f"that decomposes over current COMPONENTS rather than postsynaptic populations, "
            f"so its numbers are not the quantity Fig. 2C plots. Regenerate it (issue #6)."
        )

    firing = [row for row in firing if int(row["n"]) in set(shared)]
    current = [row for row in current if int(row["n"]) in set(shared)]

    sizes, r_bar = _mean_by_size(firing, "r_EE")
    components = {name: _mean_by_size(current, name)[1]
                  for name in ("c_EE", "c_II", "c_XX", "c_EI", "c_EX", "c_IX", "c_total")}

    for name, series in [("r_EE", r_bar)] + list(components.items()):
        if not np.isfinite(series).all():
            raise ValueError(
                f"{name} is non-finite at sizes {list(sizes[~np.isfinite(series)])}. A log "
                f"axis would drop those points silently. Most likely a neuron never changed "
                f"state, making np.corrcoef return NaN for its whole row (block2/eval.py)."
            )

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, colour in (("c_EE", GREEN), ("c_II", RED), ("c_XX", BLUE),
                         ("c_EI", ORANGE), ("c_EX", "#7b3294"), ("c_IX", "#8c6d31")):
        _plot_signed(ax, sizes, components[name], colour, _label(name))
    _plot_signed(ax, sizes, components["c_total"], BLACK, r"$c$ (total)")
    _plot_signed(ax, sizes, r_bar, BLACK, r"$\bar{r}$", hollow=True)

    c_total = components["c_total"]
    # Guide lines anchored at the smallest N so they sit on the data rather than floating.
    ax.loglog(sizes, np.abs(c_total)[0] * np.sqrt(sizes[0] / sizes), "--", color="gray", linewidth=1)
    ax.loglog(sizes, np.abs(r_bar)[0] * (sizes[0] / sizes), ":", color="gray", linewidth=1)
    # ha="right" keeps both labels inside the axes; the default left alignment puts them
    # past the right spine, where they are clipped.
    ax.annotate(r"$1/\sqrt{N}$", (sizes[-1], np.abs(c_total)[0] * np.sqrt(sizes[0] / sizes[-1])),
                color="gray", fontsize=9, ha="right", va="top")
    ax.annotate(r"$1/N$", (sizes[-1], np.abs(r_bar)[0] * sizes[0] / sizes[-1]),
                color="gray", fontsize=9, ha="right", va="top")

    ax.set_xlabel("Network size N")
    ax.set_ylabel("Correlation coefficient")
    ax.set_title("Fig. 2C")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
