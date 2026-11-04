"""Shared set-up of the figure scripts: the runs, the data readers and the style.

- The three glaciers, their Newton-CG reference run and their four network runs (one per number N of iterations
  per retraining after the first year).
- Readers of data/ (written by ../runs/export_figure_data.py from the model runs): the maps, the time series, the per-frame
  errors, the computing times.
- The style: 7 pt text at the print size of a Copernicus figure (17 cm wide), the fixed colour per N, the ink and
  chrome of the dataviz reference palette (light).

Not a script: imported by the fig_gmd_*.py scripts.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FIGURES = HERE  # the figures are written next to the scripts, under the file names of the manuscript

REF = os.environ.get("GMD_REF_NAME", "reference")  # name of the reference run (identity mapping solved with Newton-CG)
REF_MAP = os.environ.get("GMD_REF_MAP_NAME", "Reference")  # legend name of the reference run, paired with "Neural network"
NET = "Neural network"  # legend name of the neural-network runs
NBIT_SHOWN = 20  # the representative retraining budget shown where only one network run is drawn

NBITS = [5, 10, 20, 50]  # iterations per retraining after the first year
ACCENT_ORANGE, ACCENT_BLUE = "#eb6834", "#2a78d6"  # the two accents of all figures (also Figs. 2 and 3)
COLOUR = {5: "#9e2a10", 10: ACCENT_ORANGE, 20: ACCENT_BLUE, 50: "#1b3f73"}  # oranges for small N, blues for large N (N = 20, the run shown alone, in the accent blue); colour-vision-deficiency safe

# ink and chrome: dataviz reference palette, light mode
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
GREY_TRACE, GREY_INIT = "#d4d3cd", "#8a8983"  # one-off costs: start-up, initial training (hatched)
EXACT_LS = (0, (4, 2))  # Newton-CG: dashed ink, drawn on top
UNIT = r"m yr$^{-1}$"

MM = 1.0 / 25.4
W_FULL = 170 * MM
FONT = 7.0  # pt at print size, everywhere (GMD: >= 7 pt)

GLACIERS = [
    dict(
        key="jak",
        name="Sermeq Kujalleq",
        region="Greenland",
        dx="150 m",
        years=25.0,
        exact="glacier_jak_P1_30_exact_fast_mt5_25yr",
        runs={n: f"j25_n25_{n}_mt5" for n in NBITS},
    ),
    dict(
        key="pig",
        name="Pine Island Glacier",
        region="Antarctica",
        dx="1 km",
        years=100.0,
        exact="glacier_pig_exact_fast_mt5_100yr",
        runs={n: f"pig_n50_{n}_mt5_100yr" for n in NBITS},
    ),
    dict(
        key="men",
        name="Mendenhall Glacier",
        region="Alaska",
        dx="50 m",
        years=25.0,
        exact="glacier_men_exact_fast_mt5_25yr",
        runs={n: f"m25_n50_{n}_mt5" for n in NBITS},
    ),
]


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
    matplotlib.rcParams["hatch.color"] = "white"
    matplotlib.rcParams["hatch.linewidth"] = 0.6
    """rcParams of the paper figures, 7 pt."""
    family = font_family()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": FONT,
            "axes.titlesize": FONT,
            "axes.labelsize": FONT,
            "xtick.labelsize": FONT,
            "ytick.labelsize": FONT,
            "legend.fontsize": FONT,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
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
            "axes.titlepad": 4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.handlelength": 2.0,
            "legend.columnspacing": 1.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 150,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.default": "regular",
        }
    )


def letter(fig, ax, k, dx_mm=0.0, dy_mm=0.0):
    """Panel letter (a), (b), ... at the top left corner of ax, shifted by (dx_mm, dy_mm)."""
    pos = ax.get_position()
    w, h = fig.get_size_inches()
    fig.text(
        pos.x0 + dx_mm * MM / w,
        pos.y1 + dy_mm * MM / h,
        f"({'abcdefghij'[k]})",
        ha="right",
        va="bottom",
        fontsize=FONT + 1,
        fontweight="bold",
        color=INK,
    )


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def read_csv(name):
    return pd.read_csv(os.path.join(DATA, name), float_precision="round_trip", keep_default_na=True)


# ---- data readers ------------------------------------------------------------------------------------------------


def maps(g):
    """The map data of glacier g (data/maps_KEY.nc)."""
    return xr.open_dataset(os.path.join(DATA, f"maps_{g['key']}.nc"))


def state(ds, name):
    """The fields of one state of a map file ('exact_t0', 'exact_end', 'net20_end'), in float64."""
    return {
        v: np.asarray(ds[f"{name}_{v}"].values, dtype="float64") for v in ("thk", "topg", "water_level", "velsurf_mag")
    }


def keep_mask(ds):
    """Inside the scored domain (False on PIG's quasi-rigid band)."""
    return ds.keep.values > 0


_TS = None


def timeseries(run):
    """The marine_ts time series of a run (grounding line s_gl and calving front s_cf along the flowline, VAF, ...)."""
    global _TS
    if _TS is None:
        _TS = read_csv("timeseries.csv")
    return _TS[_TS.run == run].reset_index(drop=True)


def errors(g):
    """Per-frame errors of the glacier's network runs against Newton-CG, one row per (run, t)."""
    err = read_csv("errors.csv")
    return err[err.run.isin(list(g["runs"].values()))]


def time_mean(t, v):
    """Mean uniform in time (linear between frames)."""
    t, v = np.asarray(t, float), np.asarray(v, float)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(v, t) / (t[-1] - t[0]))


def mean_errors(g, err=None):
    """Per N: time-mean relative RMSE (%), time-mean RMSE (m/yr), time-mean RMS speed of Newton-CG (m/yr),
    the last frame scored and the final grounding-line / VAF differences."""
    err = errors(g) if err is None else err
    rows = {}
    for n, run in g["runs"].items():
        d = err[err.run == run].sort_values("t")
        d = d[np.isfinite(d.r_all) & np.isfinite(d.rmse_all) & (d.r_all > 0)]
        if len(d) < 2:
            continue
        rows[n] = dict(
            r_mean=time_mean(d.t, d.r_all),
            rmse_mean=time_mean(d.t, d.rmse_all),
            rms_speed=time_mean(d.t, 100 * d.rmse_all / d.r_all),
            t_end=float(d.t.iloc[-1]),
            r_max_late=float(d.r_all[d.t > 1.0 + 1e-9].max()),
            r_end=float(d.r_all.iloc[-1]),
            d_gl_end=float(d.d_gl.iloc[-1]),
            d_vaf_end=float(d.d_vaf.iloc[-1]),
            d_gl_absmax=float(np.nanmax(np.abs(d.d_gl))),
            frames=len(d),
        )
    return rows


INCLUDE_DIRECT = False  # count the direct Newton-CG solve of the stopping criterion in the emulator's cost?


def costs(g):
    """Per N: the computing-time parts (min: start-up, [direct solve,] initial training, time stepping), the
    total and the speed-up against the Newton-CG run's wall time; and that wall time (min). The direct solve of
    the stopping criterion is left out of the parts and of the total unless INCLUDE_DIRECT; it is returned
    separately as 'direct' (min)."""
    tab = read_csv("costs.csv")
    exact_min = float(tab[tab.run == g["exact"]].wall_min.iloc[0])
    rows = {}
    for n, run in g["runs"].items():
        r = tab[tab.run == run].iloc[0]
        if INCLUDE_DIRECT:
            parts = np.array([r.startup_min, r.direct_min, r.training_min, r.stepping_min], dtype=float)
        else:
            parts = np.array([r.startup_min, r.training_min, r.stepping_min], dtype=float)
        rows[n] = dict(
            parts=parts,
            direct=float(r.direct_min),
            total=float(parts.sum()),
            speed_up=exact_min / float(parts.sum()),
            finished=True,
        )
    return rows, exact_min


def vaf_ref(g):
    """VAF change of the Newton-CG run over the whole run (km3), for relative VAF differences."""
    ts = timeseries(g["exact"])
    return float(ts.vaf.iloc[-1] - ts.vaf.iloc[0]) / 1e9


def save(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"{name}.{ext}"), dpi=600 if ext == "png" else None)
    print(f"written {name}.pdf/.png")
