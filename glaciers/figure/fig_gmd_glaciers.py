#!/usr/bin/env python3
"""Figure 1: the three glaciers and their grounding-line retreat, network against Newton-CG.

  (a-c) surface speed at t = 0 of the Newton-CG run (turbo on a log scale from 1 m/yr; land, open water and the
        black ice margin in fixed colours), Newton-CG's grounding line at t = 0 (thick dark green) and at the end of
        the run (black), both solid and without casing, and the flowline (white dashes); the subtitles give the grid
        spacing and size (nx x ny). The maps show no network run: (d-f) compare them;
  (d-f) change of the grounding-line position along that flowline (s_gl of data/timeseries.csv) against time: the
        network runs (one colour per number N of iterations per retraining after the first year) and Newton-CG
        (dashed, on top).

Columns: Sermeq Kujalleq (150 m, 25 yr), Pine Island Glacier (1 km, 100 yr), Mendenhall Glacier (50 m, 25 yr). The
grounding line is drawn where the height above flotation changes sign between grounded ice and floating ice or open
water (the grounded front of a tidewater or lake-terminating glacier included), and the ice margin between ice and
everything else. PIG's quasi-rigid band (keep = 0 in data/maps_pig.nc) is drawn as land.

Data: data/maps_KEY.nc, data/timeseries.csv.  Run: python fig_gmd_glaciers.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from cmcrameri import cm as cmc  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402
from matplotlib.legend_handler import HandlerBase  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

MM = C.MM
# speed and background: lapaz (Crameri; dark to light blue, then beige for the fastest ice; perceptually uniform and
# colour-vision-deficiency safe), log scale from 1 m/yr
_SPEED_NAME = os.environ.get("GMD_SPEED_CMAP", "lapaz")
SPEED = getattr(cmc, _SPEED_NAME) if hasattr(cmc, _SPEED_NAME) else plt.get_cmap(_SPEED_NAME)
SPEED_NORM = LogNorm(1.0, 1.0e4)  # m/yr, one scale for the three glaciers
LAND, WATER, MARGIN, NODATA = "#c9c5bd", "#a8cadd", "#000000", "#e4e4e4"
# map windows (km, in the projected coordinates of each input: EPSG:3413, EPSG:3031, UTM 8N), scale bars (km)
VIEW = {
    "jak": dict(x=(-195.4, -145.4), y=(-2295.0, -2255.0), bar=10.0),
    "pig": dict(x=(-1745.0, -1465.0), y=(-415.0, -110.0), bar=50.0),
    "men": dict(x=(522.4, 537.6), y=(6475.2, 6494.0), bar=2.0),
}
# Newton-CG's grounding lines, solid and without casing: at t = 0 thick dark green, at the end black (thicker than
# the black ice margin); the flowline in white dashes
GL0 = dict(color=C.ACCENT_ORANGE, lw=1.6, ls="-")  # initial grounding line (shared by both runs): accent orange
GL_END = dict(color=C.INK, lw=1.6, ls="-")  # final grounding line of the reference run: black, as the reference everywhere
LW_MARGIN = 0.5
FLOW = dict(color=C.INK, lw=1.3, ls=(0, (0.1, 1.8)), dash_capstyle="round")  # black dots
FLOW_UNDER = dict(color="#1b1b1b", lw=1.1, ls="-")  # under the dashes: orange and black alternate, visible for all vision types


def fields(f, rho_i, rho_w, keep=None):
    """Ice, ice-free water and height above flotation of the state f (common.state); keep: the domain mask (PIG's
    quasi-rigid band is drawn as land)."""
    thk = f["thk"]
    topg = f["topg"]
    wl = f["water_level"] if "water_level" in f else np.zeros_like(thk)
    ice = thk > 0
    if keep is not None:
        ice &= keep
    water = ~ice & (topg < wl)  # ice-free ocean or lake (dry cells carry a water level below the bed)
    haf = thk - rho_w / rho_i * (wl - topg)  # height above flotation: > 0 grounded (S.masks' criterion)
    return ice, water, haf


def grounding_field(f, rho_i, rho_w, keep=None):
    """Height above flotation on ice and ice-free water, NaN on ice-free land: its zero contour is the
    grounding line, against floating ice or open water (never along the land margin)."""
    ice, water, haf = fields(f, rho_i, rho_w, keep)
    return np.where(ice | water, haf, np.nan)


def flowline(ds):
    """The flowline of the time series (km)."""
    return ds.flowline_x.values / 1e3, ds.flowline_y.values / 1e3


def scale_bar(fig, ax, km):
    """Scale bar below the map, at its left edge."""
    pos = ax.get_position()
    x0, x1 = ax.get_xlim()
    w = pos.width * km / (x1 - x0)
    fh = fig.get_size_inches()[1]
    y = pos.y0 - 2.2 * MM / fh
    fig.add_artist(
        Line2D([pos.x0, pos.x0 + w], [y, y], color=C.INK, lw=1.4, solid_capstyle="butt", transform=fig.transFigure)
    )
    fig.text(
        pos.x0 + w + 1.2 * MM / fig.get_size_inches()[0],
        y,
        f"{km:g} km",
        ha="left",
        va="center",
        color=C.INK,
        fontsize=C.FONT,
    )


def map_panel(ax, g):
    ds = C.maps(g)
    rho_i, rho_w = ds.attrs["rho_i_exact"], ds.attrs["rho_w_exact"]
    keep = C.keep_mask(ds)
    x, y = ds.x.values / 1e3, ds.y.values / 1e3
    f0 = C.state(ds, "exact_t0")
    ice, water, _ = fields(f0, rho_i, rho_w, keep)
    ax.pcolormesh(
        x,
        y,
        np.where(water, 1, 0),
        cmap=ListedColormap([LAND, WATER]),
        vmin=0,
        vmax=1,
        shading="nearest",
        rasterized=True,
    )
    speed = f0["velsurf_mag"]
    im = ax.pcolormesh(
        x,
        y,
        np.where(ice, np.maximum(speed, 1.0), np.nan),
        cmap=SPEED,
        norm=SPEED_NORM,
        shading="nearest",
        rasterized=True,
    )
    nodata = ice & ~np.isfinite(speed)
    if nodata.any():
        ax.pcolormesh(
            x, y, np.where(nodata, 1.0, np.nan), cmap=ListedColormap([NODATA]), shading="nearest", rasterized=True
        )
    ax.contour(x, y, ice.astype(float), levels=[0.5], colors=[MARGIN], linewidths=LW_MARGIN, zorder=4)
    sx, sy = flowline(ds)
    ax.plot(sx, sy, zorder=4.6, **FLOW)
    ax.contour(
        x,
        y,
        grounding_field(f0, rho_i, rho_w, keep),
        levels=[0.0],
        colors=[GL0["color"]],
        linewidths=GL0["lw"],
        linestyles=[GL0["ls"]],
        zorder=5,
    )
    f1 = C.state(ds, "exact_end")
    ax.contour(
        x,
        y,
        grounding_field(f1, rho_i, rho_w, keep),
        levels=[0.0],
        colors=[GL_END["color"]],
        linewidths=GL_END["lw"],
        linestyles=[GL_END["ls"]],
        zorder=6,
    )
    v = VIEW[g["key"]]
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


def trajectory_panel(ax, g):
    series = [(g["runs"][C.NBIT_SHOWN], C.COLOUR[C.NBIT_SHOWN], "-", 2)]
    series.append((g["exact"], C.INK, C.EXACT_LS, 4))
    for run, colour, ls, z in series:
        ts = C.timeseries(run)
        ax.plot(ts.t, (ts.s_gl - ts.s_gl.iloc[0]) / 1e3, color=colour, lw=1.0, ls=ls, zorder=z)
    ax.axhline(0.0, color=C.AXIS, lw=0.5, zorder=1)
    ax.set_xlim(0, g["years"])
    ax.set_xticks(np.arange(0, g["years"] + 1, 5 if g["years"] <= 30 else 25))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel("Time (yr)")
    C.grid(ax)


def subtitle_forms(fig, widths, gap):
    """Per glacier: region, grid spacing, grid size (nx x ny, from the map file) and length,
    in the longest of a few forms that fits above every map on one line (the same form for the three)."""
    sizes = []
    for g in C.GLACIERS:
        with C.maps(g) as ds:
            sizes.append((ds.sizes["x"], ds.sizes["y"]))
    forms = [
        lambda g, nx, ny: f"{g['region']}, {g['dx']} grid ({nx} × {ny}), {g['years']:g} yr",
        lambda g, nx, ny: f"{g['region']}, {g['dx']}, {nx} × {ny} cells, {g['years']:g} yr",
        lambda g, nx, ny: f"{g['region']}, {g['dx']} ({nx} × {ny}), {g['years']:g} yr",
        lambda g, nx, ny: f"{g['dx']}, {nx} × {ny} cells, {g['years']:g} yr",
        lambda g, nx, ny: f"{g['dx']}, {nx} × {ny} cells",
    ]
    renderer = fig.canvas.get_renderer()
    room = [w + gap - 1.0 for w in widths[:-1]] + [widths[-1] + 1.5]  # the last one: up to the colour bar's label
    for form in forms:
        texts = [form(g, nx, ny) for g, (nx, ny) in zip(C.GLACIERS, sizes)]
        fits = True
        for text, r in zip(texts, room):
            t = fig.text(0, 0, text, fontsize=C.FONT)
            width_mm = t.get_window_extent(renderer).width / fig.dpi * 25.4
            t.remove()
            fits &= width_mm <= r
        if fits:
            return texts
    return texts


class OnMap(HandlerBase):
    """Legend key of a map line: the line(s) over a swatch of the speed field, so that white lines read on paper."""

    def __init__(self, face):
        super().__init__()
        self.face = face

    def create_artists(self, legend, orig, xdescent, ydescent, width, height, fontsize, trans):
        lines = orig if isinstance(orig, tuple) else (orig,)
        artists = [
            Rectangle(
                (-xdescent, -ydescent - 0.15 * height),
                width,
                1.3 * height,
                facecolor=self.face,
                edgecolor="none",
                transform=trans,
            )
        ]
        y = -ydescent + 0.5 * height
        for ln in lines:
            a = Line2D([-xdescent + 0.08 * width, -xdescent + 0.92 * width], [y, y])
            a.update_from(ln)
            a.set_transform(trans)
            artists.append(a)
        return artists


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    C.style()
    keys = [g["key"] for g in C.GLACIERS]
    aspect = {k: (VIEW[k]["x"][1] - VIEW[k]["x"][0]) / (VIEW[k]["y"][1] - VIEW[k]["y"][0]) for k in keys}

    # layout in mm at print size: maps of one height and the colour bar, their scale bars and legend, then the
    # trajectories and the legend of the runs
    W, H_MAP = 170.0, 47.5
    left, gap, cbar_gap, cbar_w = 1.0, 4.5, 2.5, 2.4
    widths = [aspect[k] * H_MAP for k in keys]
    top_title, map_legend_h, legend_h = 8.5, 11.0, 7.0
    traj_h, traj_gap = 30.0, 6.0  # trajectory panels; the space above them (their letters)
    H = top_title + H_MAP + map_legend_h + traj_gap + traj_h + 8.5 + legend_h
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x0, y0, w, h):  # mm from the bottom left -> figure fraction
        return [x0 / W, y0 / H, w / W, h / H]

    y_map = H - top_title - H_MAP
    subtitles = subtitle_forms(fig, widths, gap)
    spare = W - left - sum(widths) - 2 * gap - cbar_gap - cbar_w - 14.0  # 14 mm: colour-bar ticks and label
    x = left + max(spare, 0.0) / 2
    maps = []
    for k, (g, w) in enumerate(zip(C.GLACIERS, widths)):
        ax = fig.add_axes(box(x, y_map, w, H_MAP))
        im = map_panel(ax, g)
        scale_bar(fig, ax, VIEW[g["key"]]["bar"])
        maps.append(ax)
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
        fig.text(
            x / W,
            (y_map + H_MAP + 1.1) / H,
            subtitles[k],
            ha="left",
            va="bottom",
            fontsize=C.FONT,
            color=C.INK2,
        )
        x += w + gap
    x_cbar = x - gap + cbar_gap
    cax = fig.add_axes(box(x_cbar, y_map, cbar_w, H_MAP))
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor(C.INK2)
    cb.ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    cb.set_label(f"Surface speed at t = 0 ({C.UNIT})", color=C.INK)

    # the map symbols, right under the maps (beside the scale bars)
    face = SPEED(SPEED_NORM(3000.0))  # a swatch of the speed field under the keys
    keys_map = [Line2D([], [], **GL0), Line2D([], [], **GL_END), Line2D([], [], **FLOW)]
    labels = ["Grounding line at t = 0", f"Grounding line at the end ({C.REF_MAP.lower()})", "Flowline of (d–f)"]
    swatch = OnMap(face)
    fig.legend(
        keys_map,
        labels,
        loc="center",
        bbox_to_anchor=(0.5, (y_map - 7.2) / H),
        ncol=3,
        handlelength=2.6,
        columnspacing=1.5,
        handletextpad=0.6,
        borderaxespad=0.0,
        handler_map={tuple: swatch, Line2D: swatch},
    )

    # trajectories: three equal panels
    y_traj = y_map - map_legend_h - traj_gap - traj_h
    t_left, t_right, t_gap = 15.0, W - 2.0, 13.0
    tw = (t_right - t_left - 2 * t_gap) / 3
    trajs = []
    for k, g in enumerate(C.GLACIERS):
        ax = fig.add_axes(box(t_left + k * (tw + t_gap), y_traj, tw, traj_h))
        trajectory_panel(ax, g)
        trajs.append(ax)
        C.letter(fig, ax, 3 + k, dx_mm=-1.0 if k else -8.0, dy_mm=2.0)
    trajs[0].set_ylabel("Grounding-line change\nalong the flowline (km)")

    runs = [Line2D([], [], color=C.COLOUR[C.NBIT_SHOWN], lw=1.2)]
    names = [f"{C.NET} (N = {C.NBIT_SHOWN})"]
    runs.append(Line2D([], [], color=C.INK, lw=1.0, ls=C.EXACT_LS))
    names.append(C.REF_MAP)
    fig.legend(
        runs,
        names,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0 / H),
        ncol=5,
        handlelength=2.2,
        columnspacing=1.6,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    C.save(fig, "fig_04_glaciers_grounding_lines")


if __name__ == "__main__":
    main()
