#!/usr/bin/env python3
"""Solver-scaling figure: cold Newton-CG solves (identity mapping) on Pine Island Glacier.

The t = 0 geometry at 4, 2, 1 and 0.5 km; MOLHO and SSA; float32 and float64; one RTX 4090; compilation excluded.
  (a) computing time of one cold solve against the number of DOFs (medians of 5 solves, min-max bands);
  (b) peak GPU memory against the number of DOFs, with the 24 GiB of the card.
Only valid timings are drawn: converged and within 1 % of the float64 reference (data/solver_scaling.csv column
`plotted`; this leaves out the out-of-memory 500 m MOLHO cases).
Encoding: MOLHO in ink, SSA in the palette blue #2a78d6; float64 solid with filled circles, float32 dashed with
hollow circles.
Self-contained (matplotlib, numpy, pandas). Run: python make_figure.py
Writes fig_07_solver_scaling.{pdf,png} next to this script.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "solver_scaling.csv")
OUT = os.path.join(HERE, "fig_07_solver_scaling")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COL = {"MOLHO": INK, "SSA": "#2a78d6"}
MM = 1.0 / 25.4
FONT = 7.0
MS = 4.0
GPU_GIB = 24564 / 1024.0  # RTX 4090 memory (nvidia-smi memory.total)
SERIES = [(b, p) for b in ("MOLHO", "SSA") for p in ("float64", "float32")]


def font_family():
    """Nimbus Sans (a Helvetica clone) when installed, otherwise Helvetica, Arial or DejaVu Sans."""
    for name in ("Nimbus Sans", "Helvetica", "Arial"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


def style():
    """The paper's rcParams (Nimbus Sans 7 pt, thin dark chrome)."""
    family = font_family()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def style_of(basis, prec):
    filled = prec == "float64"
    return dict(
        color=COL[basis],
        ls="-" if filled else (0, (3.5, 2)),
        lw=1.1,
        marker="o",
        ms=MS + (0.6 if filled else 0.0),
        mfc=COL[basis] if filled else "white",
        mec="white" if filled else COL[basis],
        mew=0.7 if filled else 1.0,
    )


def log_axes(ax):
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, numticks=12))
        axis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        axis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", color=GRID, lw=0.5)
    ax.grid(True, which="minor", color=GRID, lw=0.3, alpha=0.6)


LAYOUT = os.environ.get("GMD_SCALING_LAYOUT", "column")  # "column" (83 mm, stacked) or "wide" (120 mm, side by side)


def main():
    style()
    df = pd.read_csv(CSV)
    if LAYOUT == "column":  # one column (83 mm): the two panels stacked
        fig = plt.figure(figsize=(83 * MM, 112 * MM))
        axs = [fig.add_axes([0.17, 0.585, 0.80, 0.385]), fig.add_axes([0.17, 0.165, 0.80, 0.385])]
    else:  # 120 mm: the two panels side by side
        fig = plt.figure(figsize=(120 * MM, 60 * MM))
        axs = [fig.add_axes([0.115, 0.31, 0.355, 0.64]), fig.add_axes([0.625, 0.31, 0.355, 0.64])]
    handles = []
    for basis, prec in SERIES:
        ok = df[(df.basis == basis) & (df.precision == prec) & df.plotted].sort_values("dofs")
        st = style_of(basis, prec)
        if len(ok) > 1:
            axs[0].fill_between(ok.dofs, ok.cold_min_s, ok.cold_max_s, color=COL[basis], alpha=0.15, lw=0)
        (h,) = axs[0].plot(ok.dofs, ok.cold_median_s, **st, label=f"{basis}, {prec}")
        handles.append(h)
        axs[1].plot(ok.dofs, ok.peak_gpu_gib, **st)
    for ax in axs:
        log_axes(ax)
        ax.set_xlabel("Number of DOFs")
        ax.set_xlim(2.2e4, 3.6e6)
    if LAYOUT == "column":
        axs[0].set_xlabel("")
        axs[0].tick_params(labelbottom=False)
    axs[0].set_ylim(0.6, 30.0)
    axs[0].set_ylabel("Computing time (s)")
    axs[1].set_ylim(0.012, 90.0)
    axs[1].axhline(GPU_GIB, color=INK2, lw=0.6, ls=(0, (1, 1.5)))
    axs[1].text(0.97, GPU_GIB * 1.12, "RTX 4090 (24 GiB)", transform=axs[1].get_yaxis_transform(),
                ha="right", va="bottom", fontsize=FONT - 1, color=INK2)
    axs[1].set_ylabel("Peak GPU memory (GiB)")
    for k, ax in enumerate(axs):
        ax.text(0.04, 0.97, f"({'ab'[k]})", transform=ax.transAxes, ha="left", va="top", fontsize=FONT,
                fontweight="bold", color=INK)
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.57 if LAYOUT == "column" else 0.54, 0.0),
               handlelength=3.0, columnspacing=1.2 if LAYOUT == "column" else 2.0)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=600)
    print("wrote", OUT + ".{pdf,png}")


if __name__ == "__main__":
    main()
