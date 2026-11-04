#!/usr/bin/env python3
"""Fig. 3 of the paper: MISMIP+ with IGM (1 km, Weertman friction, neural-network mapping) against the main subset of
Cornford et al. (2020).

Top row Ice1, bottom row Ice2: (a, d) grounding-line position on the centerline, (b, e) change in grounded area,
(c, f) change in volume above flotation since t = 0. Orange: melt maintained (Ice{k}r then Ice{k}rr); blue: melt
stopped at 100 yr (Ice{k}ra). Shading: range of the 24 models (light) and of the 11 Weertman models (dark).
Data: data/igm_timeseries.csv, data/ensemble_envelope.csv (written by ../runs/make_figure_data.py).
Run: python make_figure.py   (needs numpy, pandas, matplotlib). Writes fig_03_mismip_timeseries.{pdf,png}.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "fig_03_mismip_timeseries")

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
MELT_ON, MELT_OFF = "#eb6834", "#2a78d6"
W_MAIN, W_WEERT = 0.10, 0.24  # wash opacities: main subset, Weertman models
MM = 1.0 / 25.4
W_FULL = 170 * MM


def font_family():
    """Nimbus Sans (a Helvetica clone) when installed, otherwise Helvetica, Arial or DejaVu Sans."""
    for name in ("Nimbus Sans", "Helvetica", "Arial"):
        try:
            matplotlib.font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "DejaVu Sans"


FAMILY = font_family()
plt.rcParams.update({
    "font.family": FAMILY, "font.size": 7, "axes.titlesize": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold", "mathtext.default": "regular",
})

BRANCHES = (("on", MELT_ON), ("off", MELT_OFF))  # melt continued (r -> rr) / stopped (ra)
COLS = [("gl", "xGL_centre_m", 1e3, False, "Grounding line on the centerline", "Position (km)"),
        ("area", "groundedArea_m2", 1e9, True, "Grounded area", "Change (10³ km²)"),
        ("vaf", "iceVAF_m3", 1e12, True, "Volume above flotation", "Change (10³ km³)")]


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def letter(ax, k, x=-0.02, y=1.0):
    ax.text(x, y, f"({'abcdef'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)


def igm_branch(igm, k, branch, col, rel):
    ts = lambda exp: igm[igm.exp == exp]  # noqa: E731
    r = ts(f"Ice{k}r")
    d = pd.concat([r, ts(f"Ice{k}rr").iloc[1:]]) if branch == "on" else ts(f"Ice{k}ra")
    v = d[col].values
    if rel:
        v = v - r[col].iloc[0]
    return d.t_yr.values, v


def main():
    igm = pd.read_csv(os.path.join(DATA, "igm_timeseries.csv"))
    env = pd.read_csv(os.path.join(DATA, "ensemble_envelope.csv"))
    n_main = int(env[env.subset == "main"].n.iloc[0])
    n_weert = int(env[env.subset == "weertman"].n.iloc[0])

    fig, axs = plt.subplots(2, 3, figsize=(W_FULL, 100 * MM), sharex=True, layout="constrained")
    fig.get_layout_engine().set(w_pad=4 * MM, h_pad=2.5 * MM, wspace=0.02, hspace=0.04)
    for row, k in enumerate((1, 2)):
        for c, (q, col, sc, rel, title, ylab) in enumerate(COLS):
            ax = axs[row, c]
            ax.axvline(100.0, color=AXIS, lw=0.5, zorder=0)
            for branch, hue in BRANCHES:
                for subset, alpha in (("main", W_MAIN), ("weertman", W_WEERT)):
                    e = env[(env.k == k) & (env.branch == branch) & (env.quantity == q) & (env.subset == subset)]
                    ax.fill_between(e.t_yr, e["min"] / sc, e["max"] / sc, color=hue, alpha=alpha, lw=0)
                t, v = igm_branch(igm, k, branch, col, rel)
                ax.plot(t, v / sc, color=hue, lw=1.5)
            ax.set_xlim(0, 200)
            ax.set_xticks([0, 50, 100, 150, 200])
            ax.set_ylabel(ylab)
            grid(ax, "y")
            if row == 0:
                ax.set_title(title, color=INK, fontweight="bold")
            else:
                ax.set_xlabel("Time (yr)")
            letter(ax, 3 * row + c, x=-0.20 if c == 0 else -0.17)
        # direct labels on the grounding-line panels
        ax = axs[row, 0]
        for name, branch in ((f"Ice{k}rr", "on"), (f"Ice{k}ra", "off")):
            t, v = igm_branch(igm, k, branch, "xGL_centre_m", False)
            ax.annotate(name, (t[-1], v[-1] / 1e3), xytext=(-2, 5 if branch == "off" else -5),
                        textcoords="offset points", ha="right", va="bottom" if branch == "off" else "top",
                        fontsize=6.5, color=INK)
        t, v = igm_branch(igm, k, "on", "xGL_centre_m", False)
        j = np.argmin(np.abs(t - 50.0))
        ax.annotate(f"Ice{k}r", (t[j], v[j] / 1e3), xytext=(3, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.5, color=INK)
        ax.text(-0.36, 0.5, f"Ice{k}", transform=ax.transAxes, rotation=90, ha="center", va="center",
                fontsize=8, fontweight="bold", color=INK)
    igm_handle = (Line2D([], [], color=MELT_ON, lw=1.5), Line2D([], [], color=MELT_OFF, lw=1.5))
    handles = [igm_handle, Patch(facecolor=INK2, alpha=W_WEERT + W_MAIN, lw=0), Patch(facecolor=INK2, alpha=W_MAIN, lw=0)]
    labels = ["IGM", f"MISMIP+ Weertman models (n = {n_weert})", f"all MISMIP+ models (n = {n_main})"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handleheight=0.9,
               handlelength=3.2, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", pad_inches=0.02)
    print(f"written {os.path.basename(OUT)}.pdf/.png")


if __name__ == "__main__":
    main()
