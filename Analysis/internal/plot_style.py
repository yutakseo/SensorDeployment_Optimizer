from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator

DEFAULT_DPI = 600
SPINE_WIDTH = 0.8
SCI_PLOT_STYLE = {
    "figure.dpi": 300,
    "savefig.dpi": DEFAULT_DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": SPINE_WIDTH,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": SPINE_WIDTH,
    "ytick.major.width": SPINE_WIDTH,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "axes.grid": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def applySciStyle() -> None:
    plt.rcParams.update(SCI_PLOT_STYLE)


def styleAxis(axis: Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.tick_params(which="both", top=False, right=False)
