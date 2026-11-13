#!/usr/bin/env python3
"""Fig. 7 of the paper: cost of the reference (Newton-CG) solver and comparison with PISM, on the Pine Island geometry.

Cold solves (from zero velocity) on the t = 0 geometry at 4, 2, 1 and 0.5 km. IGM: Newton-CG with the identity
mapping on one RTX 4090, MOLHO and SSA, float32 and float64. PISM 2.3.1 on 32 CPU cores, float64: SSA, and
Blatter-Pattyn with two vertical levels.
  (a) higher-order models: PISM Blatter-Pattyn against IGM MOLHO, computing time against the number of velocity DOFs;
  (b) SSA in both codes;
  (c) peak GPU memory of the IGM solves, with the 24 GiB of the card.
Data: data/cpu_gpu_comparison.csv for (a) and (b) (PISM and IGM times measured in the same campaign; at 0.5 km MOLHO
with probe_batch 1), data/solver_scaling.csv for (c). Medians of five solves; compilation and set-up excluded.
Encoding: PISM in the accent orange; IGM MOLHO in ink, IGM SSA in blue; float64 solid with filled circles, float32
dashed with hollow circles.
Run: python make_figure.py   (needs numpy, pandas, matplotlib; GMD_COST_XAXIS=spacing puts the grid spacing on x).
Writes fig_07_solver_cost.{pdf,png} next to this script (one column, panels stacked); with GMD_FIG_LAYOUT=wide, the
single-column preprint layout fig_07_solver_cost_wide.{pdf,png} (panels side by side).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter, NullLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_IGM = os.path.join(HERE, "data", "solver_scaling.csv")
CSV_CMP = os.path.join(HERE, "data", "cpu_gpu_comparison.csv")
LAYOUT = os.environ.get("GMD_COST_XAXIS", "dofs")  # "dofs": x = number of DOFs, grid spacings as labels; "spacing": x = grid spacing
WIDE = os.environ.get("GMD_FIG_LAYOUT", "column") == "wide"  # "column": 83 mm, panels stacked; "wide": 177 mm, side by side
OUT = os.path.join(HERE, "fig_07_solver_cost" + ("" if LAYOUT == "dofs" else "_spacing") + ("_wide" if WIDE else ""))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
COL = {"MOLHO": INK, "SSA": "#2a78d6"}
MM = 1.0 / 25.4
FONT = 7.0
MS = 4.0
GPU_GIB = 24564 / 1024.0  # RTX 4090 memory (nvidia-smi memory.total)


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


RES_LABEL = {"4km": "4 km", "2km": "2 km", "1km": "1 km", "500m": "0.5 km"}
SPACING = {"4km": 4.0, "2km": 2.0, "1km": 1.0, "500m": 0.5}
ORANGE = "#eb6834"  # the paper's accent orange: PISM
PISM_STYLE = dict(color=ORANGE, ls="-", lw=1.1, marker="o", ms=MS + 0.6, mfc=ORANGE, mec="white", mew=0.7)


def fmt_dofs(v, _):
    return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"


def main():
    style()
    cmp = pd.read_csv(CSV_CMP)
    igm = pd.read_csv(CSV_IGM)
    igm = igm[igm.plotted == True] if "plotted" in igm else igm[igm.status == "ok"]  # noqa: E712

    if WIDE:
        fig = plt.figure(figsize=(177 * MM, 66 * MM))
        axs = [fig.add_axes([0.075 + 0.325 * k, 0.30, 0.255, 0.62]) for k in range(3)]
    else:
        fig = plt.figure(figsize=(83 * MM, 152 * MM))
        axs = [fig.add_axes([0.17, 0.705, 0.80, 0.245]), fig.add_axes([0.17, 0.425, 0.80, 0.245]),
               fig.add_axes([0.17, 0.165, 0.80, 0.225])]

    # (a) and (b): times, PISM against IGM
    for ax, key, basis, title in ((axs[0], "bp", "MOLHO", "(a)  Higher order"),
                                  (axs[1], "ssa", "SSA", "(b)  SSA")):
        s = cmp[cmp.model == key].sort_values("allocated_velocity_dofs")
        dofs = s.allocated_velocity_dofs.values
        x = dofs if LAYOUT == "dofs" else s.resolution.map(SPACING).values
        ax.plot(x, s.pism_32core_median_s, zorder=3, **PISM_STYLE)
        ax.plot(x, s.igm_rtx4090_float64_s, zorder=3, **style_of(basis, "float64"))
        ax.plot(x, s.igm_rtx4090_float32_s, zorder=3, **style_of(basis, "float32"))
        log_axes(ax)
        ax.set_ylabel("Computing time (s)")
        ax.set_ylim(0.04, 300)
        ax.text(0.02, 0.96, title, transform=ax.transAxes, ha="left", va="top", fontsize=FONT, fontweight="bold")
        if LAYOUT == "dofs":  # grid spacing of each point, labelled above (or below) the PISM point, clear of the IGM points
            for xi, yp, y64, y32, res in zip(x, s.pism_32core_median_s, s.igm_rtx4090_float64_s, s.igm_rtx4090_float32_s,
                                             s.resolution):
                above = yp >= max(y64, y32)
                ax.annotate(RES_LABEL[res], (xi, yp), xytext=(0, 7 if above else -7), textcoords="offset points",
                            ha="center", va="bottom" if above else "top", fontsize=FONT - 0.5, color=INK2)
        else:  # number of DOFs on a top axis
            top = ax.secondary_xaxis("top")
            top.set_xscale("log")
            top.set_xticks(x)
            top.xaxis.set_minor_locator(NullLocator())
            top.set_xticklabels([fmt_dofs(v, None) for v in dofs])
            top.spines["top"].set_visible(False)
            top.tick_params(length=0, pad=1.5, labelsize=FONT - 0.5, colors=INK2)
            if key == "bp":
                top.set_xlabel("Number of velocity DOFs", fontsize=FONT - 0.5, color=INK2, labelpad=3)

    # (c): IGM peak GPU memory
    ax = axs[2]
    for basis in ("MOLHO", "SSA"):
        for prec in ("float64", "float32"):
            s = igm[(igm.basis == basis) & (igm.precision == prec)].sort_values("dofs")
            x = s.dofs.values if LAYOUT == "dofs" else s.grid.map(SPACING).values
            ax.plot(x, s.peak_gpu_gib, zorder=3, **style_of(basis, prec))
    log_axes(ax)
    ax.axhline(GPU_GIB, color=INK2, lw=0.6, ls=":", zorder=2)
    ax.text(0.98, GPU_GIB * 1.25, "RTX 4090 (24 GiB)", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=FONT, color=INK2)
    ax.set_ylim(0.012, 200)
    ax.set_ylabel("Peak GPU memory (GiB)")
    ax.text(0.02, 0.96, "(c)  IGM memory", transform=ax.transAxes, ha="left", va="top", fontsize=FONT,
            fontweight="bold")

    for ax in axs:
        if LAYOUT == "dofs":
            ax.set_xlim(2.2e4, 7e6)
        else:  # grid spacing, coarse to fine from left to right
            ax.set_xlim(5.6, 0.36)
            ax.xaxis.set_major_locator(FixedLocator([4, 2, 1, 0.5]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_locator(NullLocator())
    xlab = "Number of velocity DOFs" if LAYOUT == "dofs" else "Grid spacing (km)"
    if WIDE:
        for ax in axs:
            ax.set_xlabel(xlab)
    else:
        for ax in axs[:2]:
            ax.tick_params(labelbottom=False)
        axs[2].set_xlabel(xlab)

    if WIDE:  # one legend line with all entries
        handles = [Line2D([], [], **PISM_STYLE), Line2D([], [], **style_of("MOLHO", "float64")),
                   Line2D([], [], **style_of("MOLHO", "float32")), Line2D([], [], **style_of("SSA", "float64")),
                   Line2D([], [], **style_of("SSA", "float32"))]
        labels = ["PISM, 32 CPU cores", "IGM MOLHO, float64", "IGM MOLHO, float32", "IGM SSA, float64", "IGM SSA, float32"]
        fig.legend(handles=handles, labels=labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0),
                   handlelength=2.6, columnspacing=1.4, handletextpad=0.6)
        fig.savefig(OUT + ".pdf")
        fig.savefig(OUT + ".png", dpi=600)
        print(f"written {os.path.basename(OUT)}.pdf/.png")
        return

    # legend: PISM alone on a first, centred line; the IGM entries in two columns below
    fig.legend(handles=[Line2D([], [], **PISM_STYLE)], labels=["PISM, 32 CPU cores"], loc="lower center", ncol=1,
               bbox_to_anchor=(0.56, 0.052), handlelength=2.6, handletextpad=0.6)
    handles = [Line2D([], [], **style_of("MOLHO", "float64")), Line2D([], [], **style_of("SSA", "float64")),
               Line2D([], [], **style_of("MOLHO", "float32")), Line2D([], [], **style_of("SSA", "float32"))]
    labels = ["IGM MOLHO, float64", "IGM SSA, float64", "IGM MOLHO, float32", "IGM SSA, float32"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.0),
               handlelength=2.6, columnspacing=1.0, handletextpad=0.6)
    fig.savefig(OUT + ".pdf")
    fig.savefig(OUT + ".png", dpi=600)
    print(f"written {os.path.basename(OUT)}.pdf/.png")


if __name__ == "__main__":
    main()
