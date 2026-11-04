#!/usr/bin/env python3
"""Figure 2: accuracy and cost of the online-trained network against Newton-CG on the three glaciers.

  (a-c) relative RMSE of the surface speed against the Newton-CG run at the same time (r_all of data/errors.csv:
        grounded and floating ice together, cells ice in both runs), one line per number N of iterations per
        retraining after the first year, on the frames 0, 0.01, 0.05, 0.1, 0.25, 0.5 yr and then yearly (plain
        lines, no markers, log axis), with a dashed 5 % level;
  (d)   computing time of each run as a share of the Newton-CG run's wall time, stacked: start-up, the initial
        training, then the time stepping (data/costs.csv); the direct Newton-CG solve of the stopping criterion is
        left out (common.INCLUDE_DIRECT); the number above each bar is the wall time in minutes;
  (e)   accuracy against cost: the time-mean relative RMSE (uniform in time, linear between frames) against the
        speed-up over Newton-CG, one point per run, N written at the line ends.

Data: data/errors.csv, data/costs.csv, data/timeseries.csv. Also writes data/summary.csv, the numbers behind (d, e)
and the final grounding-line / VAF differences (the table view of the figure).
Run: python fig_gmd_performance.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MM = C.MM
RMSE_YLIM = (0.1, 100.0)  # %, decades 0.1 / 1 / 10 / 100
if C.INCLUDE_DIRECT:
    PART_COLOURS = (C.GREY_TRACE, C.MUTED, C.GREY_INIT)  # start-up, direct solve, initial training
    PART_NAMES = ("Start-up", "Direct solve", "Initial training")
else:
    PART_COLOURS = (C.GREY_TRACE, C.GREY_INIT)  # start-up, initial training
    PART_NAMES = ("Start-up", "Initial training")
MARKER = {"jak": "o", "pig": "s", "men": "^"}
MARKER_SIZE = 3.8  # pt, the same for every glacier (no size encoding)
TARGET = 5.0  # %, the dashed reference level of (a-c)


def log_percent_axis(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 1.0, 10.0, 100.0]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(*RMSE_YLIM)


def error_panel(ax, g, err):
    for n, run in g["runs"].items():
        d = err[err.run == run].sort_values("t")
        d = d[np.isfinite(d.r_all) & (d.r_all > 0)]
        ax.plot(d.t, d.r_all, color=C.COLOUR[n], lw=1.0)
    ax.axhline(TARGET, color=C.INK2, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.annotate(
        f"{TARGET:g} %",
        xy=(0.0, TARGET),
        xycoords=("axes fraction", "data"),
        xytext=(2.0, 1.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=C.FONT,
        color=C.INK2,
    )
    log_percent_axis(ax)
    ax.set_xlim(0, g["years"])
    ax.set_xticks(np.arange(0, g["years"] + 1, 5 if g["years"] <= 30 else 25))
    ax.set_xlabel("Time (yr)")
    C.grid(ax)


def cost_panel(ax, fig):
    """Grouped stacked bars: per glacier, one bar per N, as % of the Newton-CG wall time."""
    width, group_gap = 0.72, 1.1
    x0, centres, ticks, ticklabels = 0.0, [], [], []
    for g in C.GLACIERS:
        rows, exact_min = C.costs(g)
        xs = x0 + np.arange(len(C.NBITS))
        for xi, n in zip(xs, C.NBITS):
            parts = 100 * rows[n]["parts"] / exact_min
            bottom = 0.0
            for k, colour in enumerate(PART_COLOURS + (C.COLOUR[n],)):
                one_off = k < len(PART_COLOURS)
                ax.bar(xi, parts[k], bottom=bottom, width=width, color=colour, edgecolor="white", linewidth=0.3,
                       hatch="////" if (one_off and colour == C.GREY_INIT) else None)
                bottom += parts[k]
            if not rows[n]["finished"]:  # still running: its time so far, faded
                ax.bar(xi, bottom, width=width, color="white", alpha=0.5, linewidth=0)
            total = rows[n]["total"]
            label = f"{total:.0f}" if total >= 10 else f"{total:.1f}"
            ax.text(
                xi,
                bottom + 1.5,
                label if rows[n]["finished"] else f"≥{label}",
                ha="center",
                va="bottom",
                fontsize=C.FONT,
                color=C.INK2,
            )
            ticks.append(xi)
            ticklabels.append(f"{n}")
        centres.append((xs.mean(), g, exact_min))
        x0 = xs[-1] + 1 + group_gap
    ax.axhline(100.0, color=C.INK, lw=0.8, ls=C.EXACT_LS, zorder=3)
    ax.text(ticks[0] - 0.5 * width, 102.0, C.REF.capitalize(), ha="left", va="bottom", fontsize=C.FONT, color=C.INK)
    ax.text(
        ticks[0] - 0.5 * width,
        125.0,
        "Above the bars: wall time (min)",
        ha="left",
        va="center",
        fontsize=C.FONT,
        color=C.INK2,
    )
    ax.set_xticks(ticks, ticklabels)
    ax.tick_params(axis="x", length=0, pad=4)
    ax.set_xlim(ticks[0] - 0.7, ticks[-1] + 0.7)
    ax.set_ylim(0, 132)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0, 25, 50, 75, 100]))
    ax.set_ylabel(f"Computing time\n(% of {C.REF})")
    C.grid(ax)
    for xc, g, exact_min in centres:  # group names under the N ticks
        ax.annotate(
            g["name"],
            xy=(xc, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -14.0),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=C.FONT,
            color=C.INK,
        )
        ax.annotate(
            f"{C.REF.capitalize()} {exact_min:.0f} min" if exact_min >= 10 else f"{C.REF.capitalize()} {exact_min:.1f} min",
            xy=(xc, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -22.5),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=C.FONT,
            color=C.INK2,
        )
    ax.annotate(
        "N =",
        xy=(ticks[0] - 0.7, 0),
        xycoords=("data", "axes fraction"),
        xytext=(-1, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=C.FONT,
        color=C.INK2,
    )


def tradeoff_panel(ax, summary):
    """Log-log: time-mean relative RMSE against speed-up; N runs from 50 (left) to 5 (right) on every line, so
    only the two ends are labelled (the men points are too close for more)."""
    ax.axvline(1.0, color=C.INK, lw=0.8, ls=C.EXACT_LS, zorder=1)
    place = {"pig": (0, 4.0, "center", "bottom"), "jak": (0, -4.5, "center", "top"), "men": (4.5, 0, "left", "center")}
    for g in C.GLACIERS:
        s = summary[summary.glacier == g["key"]].sort_values("N")
        s = s[np.isfinite(s.r_mean) & np.isfinite(s.speed_up)]
        if s.empty:
            continue
        ax.plot(s.speed_up, s.r_mean, color=C.INK2, lw=0.8, zorder=2)
        hollow = g["key"] == "pig"  # PIG's N = 50 sits on Sermeq Kujalleq's: open squares, one size for all markers
        ax.plot(
            s.speed_up,
            s.r_mean,
            ls="none",
            marker=MARKER[g["key"]],
            ms=MARKER_SIZE,
            mfc="white" if hollow else C.INK2,
            mec=C.INK2,
            mew=0.7 if hollow else 0.0,
            zorder=3 if hollow else 4,
        )
        for k, (_, r) in enumerate(s.iloc[[0, -1]].iterrows()):  # N = 5 and N = 50
            dx, dy, ha, va = place[g["key"]]
            if g["key"] == "men" and k == 1:  # N = 50: left of its point, off the dashed line
                dx, ha = -4.5, "right"
            ax.annotate(
                f"{r.N:.0f}",
                xy=(r.speed_up, r.r_mean),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=C.FONT,
                color=C.INK2,
            )
        xm = float(np.sqrt(s.speed_up.min() * s.speed_up.max()))  # glacier names away from the points
        if g["key"] == "pig":
            ax.text(xm, 1.32 * s.r_mean.max(), g["name"], ha="center", va="bottom", fontsize=C.FONT, color=C.INK)
        elif g["key"] == "jak":
            ax.text(xm, 0.70 * s.r_mean.min(), g["name"], ha="center", va="top", fontsize=C.FONT, color=C.INK)
        else:
            ax.text(
                1.35 * s.speed_up.max(),
                float(np.sqrt(s.r_mean.min() * s.r_mean.max())),
                g["name"],
                ha="left",
                va="center",
                fontsize=C.FONT,
                color=C.INK,
            )
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([1, 2, 5, 10, 20]))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10)))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.6, 40)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.5, 10)
    ax.set_xlabel(f"Speed-up over {C.REF}")
    ax.set_ylabel("Mean relative RMSE (%)")  # mean uniform in time
    C.grid(ax, axis="both")


def summary_table():
    rows = []
    for g in C.GLACIERS:
        err = C.errors(g)
        me = C.mean_errors(g, err)
        co, exact_min = C.costs(g)
        dvaf_x = C.vaf_ref(g)
        for n in C.NBITS:
            m, c = me.get(n, {}), co[n]
            rows.append(
                dict(
                    glacier=g["key"],
                    N=n,
                    run=g["runs"][n],
                    exact=g["exact"],
                    finished=c["finished"],
                    t_scored=m.get("t_end", np.nan),
                    frames=m.get("frames", 0),
                    r_mean=m.get("r_mean", np.nan),
                    r_max_after_1yr=m.get("r_max_late", np.nan),
                    r_end=m.get("r_end", np.nan),
                    rmse_mean=m.get("rmse_mean", np.nan),
                    rms_speed_exact=m.get("rms_speed", np.nan),
                    startup_min=c["parts"][0],
                    direct_min=c["direct"],
                    training_min=c["parts"][-2],
                    stepping_min=c["parts"][-1],
                    wall_min=c["total"],
                    direct_included=C.INCLUDE_DIRECT,
                    exact_min=exact_min,
                    speed_up=c["speed_up"],
                    d_gl_end_km=m.get("d_gl_end", np.nan),
                    d_gl_absmax_km=m.get("d_gl_absmax", np.nan),
                    d_vaf_end_km3=m.get("d_vaf_end", np.nan),
                    d_vaf_end_pct=100 * m.get("d_vaf_end", np.nan) / dvaf_x if dvaf_x else np.nan,
                    dvaf_exact_km3=dvaf_x,
                )
            )
    return pd.DataFrame(rows)


def main():
    C.style()
    summary = summary_table()
    summary.to_csv(os.path.join(C.DATA, "summary.csv"), index=False, float_format="%.4g")
    print(
        summary[
            [
                "glacier",
                "N",
                "r_mean",
                "rmse_mean",
                "wall_min",
                "exact_min",
                "speed_up",
                "d_gl_end_km",
                "d_vaf_end_pct",
                "t_scored",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.2f}")
    )

    W = 170.0
    legend_h, title_h, row1_h, row1_x = 7.0, 5.0, 33.0, 9.5
    gap, row2_h, row2_x, parts_h = 9.0, 42.0, 14.0, 5.0
    H = legend_h + title_h + row1_h + row1_x + gap + row2_h + row2_x + parts_h
    fig = plt.figure(figsize=(W * MM, H * MM))

    def box(x0, y0, w, h):
        return [x0 / W, y0 / H, w / W, h / H]

    left, right, t_gap = 15.0, W - 2.0, 9.0
    tw = (right - left - 2 * t_gap) / 3
    y1 = H - legend_h - title_h - row1_h
    first = None
    for k, g in enumerate(C.GLACIERS):
        ax = fig.add_axes(box(left + k * (tw + t_gap), y1, tw, row1_h), sharey=first)
        first = first or ax
        error_panel(ax, g, C.errors(g))
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_title(f"({'abc'[k]})  {g['name']}", loc="left", fontweight="bold", color=C.INK, x=-0.0)
    first.set_ylabel("Relative RMSE of\nsurface speed (%)")

    y2 = y1 - row1_x - gap - row2_h
    w_cost, w_gap = 92.0, 17.0
    ax_cost = fig.add_axes(box(left, y2, w_cost, row2_h))
    cost_panel(ax_cost, fig)
    ax_trade = fig.add_axes(box(left + w_cost + w_gap, y2, right - left - w_cost - w_gap, row2_h))
    tradeoff_panel(ax_trade, summary)
    C.letter(fig, ax_cost, 3, dx_mm=-8.0, dy_mm=2.0)
    C.letter(fig, ax_trade, 4, dx_mm=-8.0, dy_mm=2.0)

    runs = [Line2D([], [], color=C.COLOUR[n], lw=1.2) for n in C.NBITS]
    names = [f"{C.NET}, N = {n}" for n in C.NBITS]
    fig.legend(
        runs,
        names,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0 - 0.8 / H),
        ncol=4,
        handlelength=2.2,
        columnspacing=1.8,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    pos = ax_cost.get_position()
    fig.legend(
        [Patch(facecolor=c, edgecolor="white", lw=0, hatch="////" if c == C.GREY_INIT else None) for c in PART_COLOURS] + [Patch(facecolor=C.COLOUR[20], lw=0)],
        list(PART_NAMES) + ["Time stepping (color of N)"],
        loc="lower center",
        bbox_to_anchor=(pos.x0 + 0.5 * pos.width, 0.6 / H),
        ncol=4,
        handlelength=1.0,
        handleheight=0.9,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0,
    )
    C.save(fig, "fig_05_accuracy_and_cost")


if __name__ == "__main__":
    main()
