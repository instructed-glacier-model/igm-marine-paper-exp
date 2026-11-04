#!/usr/bin/env python3
"""Fig. 2 of the paper: calving-front verification on the unconfined ice-shelf flowline of Albrecht et al. (2011).

(a, b) steady thickness and velocity profiles at 2.5 km against the analytic solution (Weertman, 1957);
(c) the steady front against the grid spacing, with the analytic 144.45 km; (d) the advance under the 250-m
minimum-thickness criterion against Eq. (13) of the paper; (e) prescribed front velocities; (f) von Mises calving
against dx/dt = u - c on the analytic profile. Orange: sub-grid front; blue: level-set front; dashed: references.
Data: data/*.csv (written by ../runs/make_figure_data.py).  Run: python make_figure.py   (needs numpy, pandas,
matplotlib). Writes fig_02_albrecht_verification.{pdf,png} next to this script.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "fig_02_albrecht_verification")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
SUB_GRID, LEVEL_SET = "#eb6834", "#2a78d6"
LW_SG, LW_LS, LW_REF = 1.8, 0.9, 0.8
METHODS = (("sub_grid", SUB_GRID, LW_SG), ("level_set", LEVEL_SET, LW_LS))
ADVANCE, RETREAT = 200.0, 300.0  # m/yr, the prescribed front velocities of (e)
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
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.handlelength": 1.8, "legend.columnspacing": 1.4,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
    "mathtext.fontset": "custom", "mathtext.rm": FAMILY, "mathtext.it": f"{FAMILY}:italic",
    "mathtext.bf": f"{FAMILY}:bold", "mathtext.default": "regular",
})


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def letter(ax, k, x=-0.02, y=1.0):
    ax.text(x, y, f"({'abcdef'[k]})", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, fontweight="bold", color=INK)


def reference(ax, x, y):
    """Dashed ink reference, drawn over the model lines: the dash gaps keep the model colour visible where the two
    coincide."""
    ax.plot(x, y, color=INK, lw=LW_REF, ls=(0, (2.4, 2.0)), zorder=4)


def along_label(ax, text, t_at, dy, t, y):
    """Label following the local slope of the curve (t, y), dy (km) off the line."""
    j = int(np.argmin(np.abs(t - t_at)))
    j0, j1 = max(j - 3, 0), min(j + 3, len(t) - 1)
    angle = np.degrees(np.arctan2(y[j1] - y[j0], t[j1] - t[j0]))
    ax.text(t[j], y[j] + dy, text, rotation=angle, rotation_mode="anchor", transform_rotates_text=True,
            ha="center", va="center", fontsize=6.5, color=INK)


def trajectory(traj, run):
    d = traj[traj.run == run]
    return d.t_yr.to_numpy(), d.x_front_km.to_numpy(), d.reference_km.to_numpy()


def main():
    profiles = pd.read_csv(os.path.join(DATA, "profiles_2.5km.csv"))
    steady = pd.read_csv(os.path.join(DATA, "steady_front.csv"))
    traj = pd.read_csv(os.path.join(DATA, "trajectories.csv"))
    x_front = float(steady.analytic_km.iloc[0])

    fig = plt.figure(figsize=(W_FULL, 102 * MM), layout="constrained")
    fig.get_layout_engine().set(w_pad=4 * MM, h_pad=2.5 * MM, wspace=0.02, hspace=0.06)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.25])
    ax_h = fig.add_subplot(gs[0, 0])
    ax_u = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1], sharey=ax_d)
    ax_f = fig.add_subplot(gs[1, 2], sharey=ax_d)

    # (a), (b) steady profiles at 2.5 km against the analytic solution
    a = profiles[profiles.scheme == "analytic"]
    reference(ax_h, np.r_[a.x_km.to_numpy(), x_front], np.r_[a.thickness_m.to_numpy(), 0.0])
    reference(ax_u, a.x_km.to_numpy(), a.velocity_m_yr.to_numpy())
    for method, hue, lw in METHODS:
        p = profiles[profiles.scheme == method]
        x, thk, u = p.x_km.to_numpy(), p.thickness_m.to_numpy(), p.velocity_m_yr.to_numpy()
        ax_h.plot(np.r_[x, x[-1] + 1.25], np.r_[thk, 0.0], color=hue, lw=lw, zorder=2)  # closed at the half cell
        ax_u.plot(x, u, color=hue, lw=lw, zorder=2)
    ax_h.set_title("Ice thickness", color=INK, fontweight="bold")
    ax_u.set_title("Ice velocity", color=INK, fontweight="bold")
    for ax, ylab, ymax in ((ax_h, "Thickness (m)", 620), (ax_u, "Velocity (m yr$^{-1}$)", 800)):
        ax.set_xlim(0, 200)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel(ylab)
        grid(ax, "y")

    # (c) steady front against the grid spacing
    ax = ax_c
    ax.axhline(x_front, color=INK, lw=LW_REF, ls=(0, (2.4, 2.0)), zorder=1)
    for (method, hue, _), shift in zip(METHODS, (1.0 / 1.04, 1.04)):
        s = steady[steady.scheme == method].sort_values("dx_km", ascending=False)
        ax.scatter(s.dx_km * shift, s.x_front_km, s=16, facecolor=hue, edgecolor="white", lw=0.7, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks([2.5, 5, 10], ["2.5", "5", "10"])
    ax.minorticks_off()
    ax.set_xlim(1.9, 13)
    ax.set_ylim(142.8, 154.8)
    ax.set_yticks([144, 146, 148, 150, 152, 154])
    ax.set_xlabel("Grid spacing (km)")
    ax.set_ylabel("Position (km)")
    ax.set_title("Steady front", color=INK, fontweight="bold")
    grid(ax, "y")

    # (d) thickness rule: advance from a two-cell shelf against Eq. (13)
    ax = ax_d
    t_ref, _, ref = trajectory(traj, "adv_sub_grid_5km")
    for method, hue, lw in METHODS:
        t, x, _ = trajectory(traj, f"adv_{method}_5km")
        ax.plot(t, x, color=hue, lw=lw, zorder=2)
    reference(ax, t_ref, ref)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 210)
    ax.set_ylabel("Position (km)")
    ax.set_title("Minimum thickness (250 m)", color=INK, fontweight="bold")

    # (e) ice-speed law: prescribed front velocities
    ax = ax_e
    for run, dy in (("advance", 12.0), ("retreat", -12.0)):
        t_ref, _, ref = trajectory(traj, f"{run}_sub_grid_5km")
        for method, hue, lw in METHODS:
            t, xm, _ = trajectory(traj, f"{run}_{method}_5km")
            ax.plot(t, xm, color=hue, lw=lw, zorder=2)
        reference(ax, t_ref, ref)
        speed = ADVANCE if run == "advance" else -RETREAT
        along_label(ax, f"{speed:+g} m yr$^{{-1}}$", 150.0, dy, t_ref, ref)
    ax.set_xlim(0, 300)
    ax.tick_params(labelleft=False)
    ax.set_title("Ice-speed law", color=INK, fontweight="bold")

    # (f) von Mises calving against the ODE on the analytic profile
    t_ref, _, ref = trajectory(traj, "vm_sub_grid_5km")
    for method, hue, lw in METHODS:
        t, x, _ = trajectory(traj, f"vm_{method}_5km")
        ax_f.plot(t, x, color=hue, lw=lw, zorder=2)
    reference(ax_f, t_ref, ref)
    ax_f.set_xlim(0, 300)
    ax_f.tick_params(labelleft=False)
    ax_f.set_title("Von Mises calving", color=INK, fontweight="bold")

    for k, (ax, xoff) in enumerate(((ax_h, -0.20), (ax_u, -0.17), (ax_c, -0.17), (ax_d, -0.20), (ax_e, -0.05), (ax_f, -0.05))):
        letter(ax, k, x=xoff)
    for ax in (ax_d, ax_e, ax_f):
        ax.set_xlabel("Time (yr)")
        grid(ax, "y")
    for row_ax, name in ((ax_h, "Steady state"), (ax_d, "Transient evolution")):
        row_ax.text(-0.36, 0.5, name, transform=row_ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=INK)

    handles = [Line2D([], [], color=SUB_GRID, lw=LW_SG), Line2D([], [], color=LEVEL_SET, lw=LW_LS),
               Line2D([], [], color=INK, lw=LW_REF, ls=(0, (2.4, 2.0)))]
    fig.legend(handles, ["sub-grid front", "level-set front", "reference"], loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=3)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", pad_inches=0.02)
    print(f"written {os.path.basename(OUT)}.pdf/.png")


if __name__ == "__main__":
    main()
