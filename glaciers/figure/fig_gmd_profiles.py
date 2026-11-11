#!/usr/bin/env python3
"""Fig. C1 of the paper: profiles along the flowline of each glacier: bed, ice and water at t = 0, the ice of the
reference run at the end, and the surface speed at t = 0.

One column, one panel per glacier. Left axis: elevation (m); the bed, the water and the ice at t = 0 shaded, the
ice of the reference run at the end as a dashed outline (surface, base and front). Right axis: surface speed at t = 0. Distance runs along the flowline of Fig. 4, from its
upstream end. Data: data/maps_GLACIER.nc (fields at t = 0 and at the end, and the flowline). Run: python fig_gmd_profiles.py
(needs scipy); writes fig_C1_flowline_profiles.{pdf,png}.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import map_coordinates, uniform_filter1d  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ICE, BED, WATER, SKY = "#dfe6ea", "#c9c5bd", "#a8cadd", "white"
RHO, RHO_W = {"jak": (917.0, 1023.0), "pig": (917.0, 1027.0), "men": (917.0, 1000.0)}, None


def sample(ds, name, xs, ys, order=1):
    """Sample a map field along the flowline (bilinear by default; nearest for masks and speeds)."""
    x, y = ds.x.values, ds.y.values
    fi = (xs - x[0]) / (x[1] - x[0])
    fj = (ys - y[0]) / (y[1] - y[0])
    return map_coordinates(ds[name].values, [fj, fi], order=order, mode="nearest")


def geometry(ds, tag, xs, ys, rho, rho_w):
    """Surface, base and ice mask along the flowline for one state (fields exact_<tag>_*)."""
    thk = sample(ds, f"exact_{tag}_thk", xs, ys)
    bed = sample(ds, f"exact_{tag}_topg", xs, ys)
    wl = sample(ds, f"exact_{tag}_water_level", xs, ys)
    ice = sample(ds, f"exact_{tag}_thk", xs, ys, order=0) > 1.0
    low = np.where(ice, np.maximum(bed, wl - rho / rho_w * thk), bed)
    surf = np.where(ice, low + thk, np.nan)
    return thk, bed, wl, ice, low, surf


def outline(ax, s, ice, low, surf, **kw):
    """Outline of the ice: surface, base and the vertical front(s) where the ice ends."""
    ax.plot(s, surf, **kw)
    ax.plot(s, np.where(ice, low, np.nan), **kw)
    edges = np.flatnonzero(np.diff(ice.astype(int)) != 0)
    for e in edges:
        k = e if ice[e] else e + 1
        ax.plot([s[k], s[k]], [low[k], surf[k]], **kw)


def main():
    C.style()
    fig = plt.figure(figsize=(83 * C.MM, 135 * C.MM))
    axes = []
    for k, g in enumerate(C.GLACIERS):
        ax = fig.add_axes([0.16, 0.765 - 0.29 * k, 0.69, 0.195])
        rho, rho_w = RHO[g["key"]]
        with C.maps(g) as ds:
            xs, ys = ds.flowline_x.values, ds.flowline_y.values
            s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))]) / 1e3
            thk0, bed0, wl0, ice0, low0, surf0 = geometry(ds, "t0", xs, ys, rho, rho_w)
            thk1, bed1, wl1, ice1, low1, surf1 = geometry(ds, "end", xs, ys, rho, rho_w)
            u0 = sample(ds, "exact_t0_velsurf_mag", xs, ys, order=0)
            # running mean over about 0.5 km along the flowline (the speed is sampled node by node)
            ds_km = np.median(np.diff(s))
            win = max(3, int(round(0.5 / ds_km)) | 1)
            valid = ice0 & (u0 > 0.0)  # average over ice points only, so that the front does not pull the speed down
            num = uniform_filter1d(np.where(valid, np.log(np.where(valid, u0, 1.0)), 0.0), win, mode="constant")
            den = uniform_filter1d(valid.astype(float), win, mode="constant")
            u0 = np.where(den > 0, np.exp(num / np.maximum(den, 1e-12)), np.nan)
        ymin = np.nanmin(bed0) - 0.08 * (np.nanmax(surf0) - np.nanmin(bed0))
        # water (ocean or lake, and the cavity under floating ice), bed and ice at t = 0
        water_top = np.where(ice0, np.minimum(wl0, low0), wl0)
        ax.fill_between(s, bed0, np.maximum(water_top, bed0), color=WATER, lw=0, zorder=1)
        ax.fill_between(s, ymin, bed0, color=BED, lw=0, zorder=2)
        ax.fill_between(s, np.where(ice0, low0, np.nan), surf0, color=ICE, lw=0, zorder=3)
        outline(ax, s, ice0, low0, surf0, color=C.INK, lw=0.7, zorder=5)
        # geometry of the reference run at the end: dashed outline, front included
        outline(ax, s, ice1, low1, surf1, color=C.INK, lw=0.7, ls=(0, (2.5, 1.8)), zorder=6)
        ax.set_xlim(s[0], s[-1])
        ax.set_ylim(ymin, np.nanmax(surf0) * 1.06)
        ax.set_ylabel("Elevation (m)")
        C.grid(ax)
        # surface speed at t = 0, right axis (nearest sampling, ice only: no interpolation across the front)
        ax2 = ax.twinx()
        ax2.plot(s, np.where(ice0, u0, np.nan), color=C.COLOUR[5], lw=0.7, zorder=7)
        if g["key"] == "men":  # speeds within two decades: a linear axis reads better than a log one
            ax2.set_ylim(0.0, 160.0)
        else:
            ax2.set_yscale("log")
        ax2.set_ylabel("Surface speed (m yr$^{-1}$)", color=C.COLOUR[5])
        ax2.tick_params(axis="y", colors=C.COLOUR[5])
        ax2.spines["top"].set_visible(False)
        ax.set_title(f"({'abc'[k]})  {g['name']}", loc="left", fontsize=C.FONT, fontweight="bold", pad=3)
        axes.append(ax)
    axes[-1].set_xlabel("Distance along the flowline (km)")
    handles = [Patch(facecolor=ICE, edgecolor=C.INK, lw=0.6), Patch(facecolor=BED, lw=0), Patch(facecolor=WATER, lw=0),
               Line2D([], [], color=C.INK, lw=0.7, ls=(0, (2.5, 1.8))), Line2D([], [], color=C.COLOUR[5], lw=0.7)]
    labels = ["Ice at t = 0", "Bed", "Water", "Ice at the end (reference run)", "Surface speed at t = 0"]
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.52, 0.0), handlelength=1.8,
               columnspacing=1.2, handletextpad=0.5, frameon=False)
    C.save(fig, "fig_C1_flowline_profiles")


if __name__ == "__main__":
    main()
