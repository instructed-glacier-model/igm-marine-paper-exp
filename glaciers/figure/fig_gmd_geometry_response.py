#!/usr/bin/env python3
"""Figure 8 of the paper: the emulator's response to the geometry change against Newton-CG's (the split of Rosier et al.).

(a-c) for each glacier, the network run with N = 20 iterations per retraining: one point per saved frame,
      e_geom = V_net(I^E) - V_net(I^R) (the network of time t on the network run's state against on Newton-CG's)
      against e_phys = V_ex(I^E) - V_ex(I^R) (the Newton-CG solution on the same two states), with V the mean surface
      speed over the cells with more than 1 m of ice in both states (Rosier et al., egusphere-2026-5523, Eq. 11);
      the 1:1 line is dashed.
Data: data/geometry_response.csv (../runs/geometry_response/).  Run: python fig_gmd_geometry_response.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MM = C.MM
DATA = os.path.join(C.DATA, "geometry_response.csv")
NBIT = 20  # the representative network run of the decomposition


def main():
    C.style()
    err = pd.read_csv(DATA)

    W, side, left, gap = 170.0, 40.0, 14.0, 17.0  # mm: three square panels
    top, bottom = 7.0, 13.0
    H = top + side + bottom
    fig = plt.figure(figsize=(W * MM, H * MM))
    colour = C.COLOUR[NBIT]  # the colour of N = 20, as in Figs. 4-6
    for k, g in enumerate(C.GLACIERS):
        x0 = left + k * (side + gap)
        ax = fig.add_axes([x0 / W, bottom / H, side / W, side / H])
        d = err[err.glacier == g["key"]]
        d = d[np.isfinite(d.e_geom) & np.isfinite(d.e_phys)]
        if len(d):
            lim = 1.1 * float(np.max(np.abs(np.r_[d.e_geom.values, d.e_phys.values])))
            ax.axhline(0.0, color=C.GRID, lw=0.5, zorder=0)
            ax.axvline(0.0, color=C.GRID, lw=0.5, zorder=0)
            ax.plot([-lim, lim], [-lim, lim], color=C.INK2, lw=0.6, ls=C.EXACT_LS, zorder=1)
            ax.plot(d.e_geom, d.e_phys, ls="none", marker="o", ms=2.6, mew=0, color=colour, zorder=2)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            locator = matplotlib.ticker.MaxNLocator(nbins=5, symmetric=True)
            ax.xaxis.set_major_locator(locator)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, symmetric=True))
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(g["name"], loc="left", fontweight="bold", color=C.INK)
        ax.set_xlabel(rf"$e_\mathrm{{geom}}$ ({C.UNIT})")
        ax.set_ylabel(rf"$e_\mathrm{{phys}}$ ({C.UNIT})")
        C.letter(fig, ax, k, dx_mm=-9.0, dy_mm=2.6)
    C.save(fig, "fig_08_geometry_response")


if __name__ == "__main__":
    main()
