"""Plotting helpers shared by all three blocks: the paper's colour code, CSV reading, and
the scale-bar style Fig. 1's trace panels use instead of full axes.

Here rather than in a block because the colour code IS shared -- the paper sets it in
Fig. 2B's legend and then says "Color code as in (B)" for 2C, 2D and 2E, and Fig. 3 reuses
the same green/red for E and I.
"""
import csv

import numpy as np

# Fig. 2B's legend, reused across Fig. 2 and Fig. 3: E green, I red, X blue, totals black.
GREEN, RED, BLUE, BLACK = "#2ca02c", "#d62728", "#1f77b4", "#000000"
ORANGE, GOLD, PURPLE = "#e08214", "#d4a017", "#7b3294"


def read_csv(path: str) -> list[dict]:
    """Rows as dicts with every value floated. Every panel's input is a numeric CSV."""
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def round_scale(value: float) -> float:
    """A visually clean scale-bar magnitude close to value (1, 2 or 5 times a power of 10)."""
    if value <= 0:
        return 1.0
    exponent = np.floor(np.log10(value))
    for m in (1, 2, 5, 10):
        if m * 10 ** exponent >= value:
            return float(m * 10 ** exponent)
    return float(10 ** (exponent + 1))


def scale_bar(ax, y_range: float, y_unit: str, x_range_ms: float = 50.0) -> None:
    """Hides the box and ticks, then draws an L-shaped scale bar bottom-right, sized to the
    data. This is the paper's own presentation for its illustrative trace panels.
    """
    ax.axis("off")
    y_bar = round_scale(y_range * 0.4)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x0 = xlim[1] - x_range_ms
    y0 = ylim[0] + 0.05 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + x_range_ms], [y0, y0], color="black", linewidth=1.5)
    ax.plot([x0, x0], [y0, y0 + y_bar], color="black", linewidth=1.5)
    ax.text(x0 + x_range_ms / 2, y0 - 0.03 * (ylim[1] - ylim[0]), f"{x_range_ms:.0f} ms",
            ha="center", va="top", fontsize=8)
    ax.text(x0 - 0.01 * (xlim[1] - xlim[0]), y0 + y_bar / 2, f"{y_bar:.2g} {y_unit}",
            ha="right", va="center", fontsize=8)
