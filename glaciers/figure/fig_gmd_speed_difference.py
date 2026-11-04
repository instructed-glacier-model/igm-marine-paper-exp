#!/usr/bin/env python3
"""Figure 3 (optional): where the network's surface speed departs from Newton-CG.

Two versions of the same three maps, network run with N = 20 iterations per retraining after the first year
(j25_n25_20_mt5, pig_n50_20_mt5_100yr, m25_n50_20_mt5) against the Newton-CG run at the end of each run, on the
cells that are ice in both:
  fig_gmd_speed_difference      the difference u_network - u_Newton-CG (m/yr), one symmetric colour scale per
                                glacier, rounded up (1-2-5) from the 95th percentile of |difference| in the map
                                window;
  fig_gmd_speed_difference_rel  the relative difference (u_network - u_Newton-CG) / (|u_Newton-CG| + u0), in %, with
                                the floor u0 = 10 m/yr (no blow-up on slow ice); one fixed symmetric scale for the
                                three glaciers, -25 to +25 % (REL_LIMIT).
Blue: network slower, red: faster; the few percent of cells beyond the scale are clipped (arrows). Land, open water
and the black ice margin (of either run) as in fig_gmd_glaciers.py; ice in one run only is pale grey, hatched (no
comparison); neither has a legend entry. Lines: the grounding line at that time, network (solid, colour of N) and
Newton-CG (dashed). The definitions of the colour bars belong in the caption.

These are trajectory differences: they include the drift of the two geometries (grounding line and front a few
cells apart), not only the emulator's error on a common geometry. Same map windows as fig_gmd_glaciers.py.

Data: data/maps_KEY.nc (states exact_end and net20_end).  Run: python fig_gmd_speed_difference.py [--abs]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import fig_gmd_glaciers as G  # noqa: E402  (map windows, colours, grounding-line field, scale bars)

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MM = C.MM
NBIT = 20  # the network run of the maps (data/maps_KEY.nc: net20_end)
U0 = 10.0  # m/yr, floor of the relative difference
# grounding lines at the time shown: the network's solid in the colour of N, Newton-CG's dashed ink on top
GL_NET = dict(lw=1.3, ls="-")
GL_EXACT = dict(color=C.INK, lw=0.9, ls=C.EXACT_LS)
REL_LIMIT = 25.0  # %, fixed scale of the relative version; None: from the pooled 95th percentile
# diverging: the blue ramp of the reference palette (steps 700 -> 100) and a red arm on the same lightness steps
# at the hue of the palette's red (#e34948), both validated monotone with equal steps, around the neutral midpoint
DIVERGING = LinearSegmentedColormap.from_list(  # blue-orange on the hues of the accent colours, with arms of equal
    # lightness (CIECAM02 J = 20, 45, 94), so that equal differences of either sign look equally strong; CVD safe
    "diff", [(0.0, "#224476"), (0.25, "#4e89d8"), (0.5, "#f4f2ef"), (0.75, "#ca6136"), (1.0, "#6c2a1a")]
)


def nice_limit(v):
    """Round up to 1, 2 or 5 times a power of ten."""
    e = 10.0 ** np.floor(np.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * e + 1e-12:
            return m * e
    return 10 * e


def compare(g):
    """Fields of the network run (N = 20) and of Newton-CG at the end of the run, on the common grid."""
    ds = C.maps(g)
    rho_i, rho_w = ds.attrs["rho_i_exact"], ds.attrs["rho_w_exact"]
    keep = C.keep_mask(ds)
    fx, fn = C.state(ds, "exact_end"), C.state(ds, "net20_end")
    ice_x, water_x, _ = G.fields(fx, rho_i, rho_w, keep)
    ice_n, water_n, _ = G.fields(fn, rho_i, rho_w, keep)
    ux, un = fx["velsurf_mag"], fn["velsurf_mag"]
    both = ice_x & ice_n & np.isfinite(ux) & np.isfinite(un)
    x, y = ds.x.values / 1e3, ds.y.values / 1e3
    v = G.VIEW[g["key"]]
    inside = (
        (x[None, :] >= v["x"][0]) & (x[None, :] <= v["x"][1]) & (y[:, None] >= v["y"][0]) & (y[:, None] <= v["y"][1])
    )
    return dict(
        x=x,
        y=y,
        fx=fx,
        fn=fn,
        keep=keep,
        rho=(rho_i, rho_w),
        rho_n=(ds.attrs["rho_i_network"], ds.attrs["rho_w_network"]),
        both=both,
        one=ice_x ^ ice_n,
        water=water_x & water_n,
        ice=ice_x | ice_n,
        inside=inside,
        abs=un - ux,
        rel=100.0 * (un - ux) / (np.abs(ux) + U0),
    )


def panel(ax, g, n, d, metric, lim):
    x, y = d["x"], d["y"]
    ax.pcolormesh(
        x,
        y,
        np.where(d["water"], 1, 0),
        cmap=ListedColormap([G.LAND, G.WATER]),
        vmin=0,
        vmax=1,
        shading="nearest",
        rasterized=True,
    )
    im = ax.pcolormesh(
        x,
        y,
        np.where(d["both"], d[metric], np.nan),
        cmap=DIVERGING,
        vmin=-lim,
        vmax=lim,
        shading="nearest",
        rasterized=True,
    )
    if d["one"].any():  # ice in one run only: no comparison (plot_floating's no-data style)
        with matplotlib.rc_context({"hatch.color": "#9a9a9a", "hatch.linewidth": 0.4}):
            ax.contourf(
                x, y, d["one"].astype(float), levels=[0.5, 1.5], colors=[G.NODATA], hatches=["//////"], zorder=2
            )
    ax.contour(x, y, d["ice"].astype(float), levels=[0.5], colors=[G.MARGIN], linewidths=G.LW_MARGIN, zorder=4)
    ax.contour(
        x,
        y,
        G.grounding_field(d["fn"], *d["rho_n"], d["keep"]),
        levels=[0.0],
        colors=[C.COLOUR[n]],
        linewidths=GL_NET["lw"],
        linestyles=[GL_NET["ls"]],
        zorder=6,
    )
    ax.contour(
        x,
        y,
        G.grounding_field(d["fx"], *d["rho"], d["keep"]),
        levels=[0.0],
        colors=[GL_EXACT["color"]],
        linewidths=GL_EXACT["lw"],
        linestyles=[GL_EXACT["ls"]],
        zorder=7,
    )
    v = G.VIEW[g["key"]]
    ax.set_xlim(*v["x"])
    ax.set_ylim(*v["y"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(C.INK2)
        ax.spines[side].set_linewidth(0.5)
    return im


def signed(v, _=None):
    return f"{v:+g}".replace("-", "−") if v else "0"


def colourbar(fig, im, rect, lim, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="both")
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(C.INK2)
    cb.ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([-lim, -lim / 2, 0, lim / 2, lim]))
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(signed))
    cb.set_label(label, color=C.INK, labelpad=1.5)


def figure(data, n, metric):
    keys = [g["key"] for g in C.GLACIERS]
    aspect = {k: (G.VIEW[k]["x"][1] - G.VIEW[k]["x"][0]) / (G.VIEW[k]["y"][1] - G.VIEW[k]["y"][0]) for k in keys}
    W, H_MAP, gap, left = 170.0, 46.0, 4.5, 1.0
    widths = [aspect[k] * H_MAP for k in keys]
    top_title, below = 8.5, 25.0
    H = top_title + H_MAP + below
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x0, y0, w, h):
        return [x0 / W, y0 / H, w / W, h / H]

    if metric == "rel":  # one scale for the three glaciers
        pooled = np.concatenate([np.abs(d["rel"][d["both"] & d["inside"]]) for d in data.values() if d])
        shared = REL_LIMIT if REL_LIMIT else nice_limit(float(np.percentile(pooled, 95)))
    x = left + max(W - left - sum(widths) - 2 * gap - 1.0, 0.0) / 2
    y_map = H - top_title - H_MAP
    im = None
    for k, (g, w) in enumerate(zip(C.GLACIERS, widths)):
        d = data[g["key"]]
        ax = fig.add_axes(box(x, y_map, w, H_MAP))
        if d is None:
            ax.text(0.5, 0.5, "no frame yet", transform=ax.transAxes, ha="center", va="center")
        else:
            if metric == "rel":
                lim = shared
            else:
                lim = nice_limit(float(np.percentile(np.abs(d["abs"][d["both"] & d["inside"]]), 95)))
            im = panel(ax, g, n, d, metric, lim)
            rms = float(np.sqrt(np.mean(d["abs"][d["both"]] ** 2)))
            print(
                f"  {metric} {g['key']}: N {n}, t {d['t']:g} yr: scale ±{lim:g}, RMSE {rms:.2f} m/yr, "
                f"|rel| median {np.median(np.abs(d['rel'][d['both']])):.2f} %, "
                f"p99 {np.percentile(np.abs(d['rel'][d['both']]), 99):.1f} %"
            )
            if metric == "abs":
                cw = min(w * 0.66, 34.0)
                colourbar(fig, im, box(x + w - cw, y_map - 6.5, cw, 2.0), lim, f"Speed difference ({C.UNIT})")
        G.scale_bar(fig, ax, G.VIEW[g["key"]]["bar"])
        fig.text(
            x / W,
            (y_map + H_MAP + 4.4) / H,
            f"({'abc'[k]})  {g['name']}",
            ha="left",
            va="bottom",
            fontsize=C.FONT,
            fontweight="bold",
            color=C.INK,
        )
        if d is not None:
            fig.text(
                x / W,
                (y_map + H_MAP + 1.1) / H,
                f"t = {d['t']:g} yr",
                ha="left",
                va="bottom",
                fontsize=C.FONT,
                color=C.INK2,
            )
        x += w + gap
    if metric == "rel" and im is not None:
        colourbar(
            fig,
            im,
            box(W / 2 - 30.0, y_map - 7.5, 60.0, 2.0),
            shared,
            "Relative speed difference (%)",
        )
    handles = [
        Line2D([], [], color=C.COLOUR[n], **GL_NET),
        Line2D([], [], **GL_EXACT),
    ]
    labels = [
        f"Grounding line, neural network (N = {n})",
        f"Grounding line, {C.REF_MAP.lower()}",
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0 / H),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    C.save(fig, "fig_06_speed_difference" if metric == "rel" else "fig_06_speed_difference_abs")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--abs", action="store_true", help="also draw the absolute difference (not in the paper)")
    args = parser.parse_args()
    C.style()
    data = {}
    for g in C.GLACIERS:
        d = compare(g)
        d["t"] = g["years"]
        data[g["key"]] = d
    for metric in (("abs", "rel") if args.abs else ("rel",)):
        figure(data, NBIT, metric)


if __name__ == "__main__":
    main()
