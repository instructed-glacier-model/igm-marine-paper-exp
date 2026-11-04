"""
Compare the runs with the analytic solutions (prep/common.py) and write results/summary.md.

For every run in outputs/ with an albrecht_ts.csv:

  adv_*    front position during the advance against Eq. 13 (shifted to the initial shelf
           volume), until the analytic front reaches the 250-m thickness; the final (steady)
           front against x(H = 250 m) = 144.45 km; the final H(x), u(x) on the middle row
           against Eqs. 11-12 behind the front (x < x_front - 2 dx);
  retreat_*, advance_*  front position against x0 -/+ W t (ice_speed law: the front moves at W);
  const_*  front position against dx/dt = u(x) - c, c = 800 m/yr, on the analytic profile;
  vm_*     the same with the von Mises rate on the analytic profile;

and, for all, the mass-budget residual (volume - volume0 - inflow + calved, relative to the
inflow) and the cost (wall time per model year, time steps).

Usage:  python analysis.py [RUN ...]   (from this folder; writes results/summary.md and summary.csv)
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")  # the run folders (make_figure_data.py can point elsewhere)
sys.path.insert(0, os.path.join(ROOT, "prep"))
import common as C  # noqa: E402
import make_experiments as E  # noqa: E402


def read_ts(run):
    return pd.read_csv(os.path.join(OUTPUTS, run, "albrecht_ts.csv"))


def resolution(run):
    for res, dx in C.RESOLUTIONS.items():
        if f"_{res}" in run:
            return res, dx
    raise ValueError(run)


def initial_row(run):
    """Initial thickness on the middle row of the run (the first saved frame)."""
    with xr.open_dataset(os.path.join(OUTPUTS, run, "output.nc")) as ds:
        thk = ds["thk"].isel(time=0).values
    return thk[thk.shape[0] // 2]


def front_reference(run, ts):
    """Analytic front position at the ts times (m), for the run's experiment."""
    t = ts["t"].to_numpy() - ts["t"].iloc[0]
    x0 = float(ts["x_front"].iloc[0])
    if run.startswith("adv_"):
        _, dx = resolution(run)
        t0 = C.analytic_start(initial_row(run), dx)
        return np.minimum(C.front(t + t0), C.position(C.H_CALVING))
    if run.startswith("retreat"):
        return x0 - E.RETREAT * t
    if run.startswith("advance"):
        return x0 + E.ADVANCE * t
    if run.startswith("const"):
        return C.front_ode(x0, t, lambda x: E.CONSTANT_RATE)
    if run.startswith("vm"):
        return C.front_ode(x0, t, lambda x: C.von_mises_rate(x, E.SIGMA_MAX))
    raise ValueError(run)


def profile_errors(run, dx):
    """RMSE (m, m/yr) and max relative error of H, u behind the final front."""
    path = os.path.join(OUTPUTS, run, "output.nc")
    if not os.path.exists(path):
        return {}
    with xr.open_dataset(path) as ds:
        frame = ds.isel(time=-1)
        mid = frame.sizes["y"] // 2
        thk = frame["thk"].values[mid]
        u = frame["ubar"].values[mid]
    xf = np.arange(len(thk)) * dx  # nodes, from the inflow boundary
    ice = np.flatnonzero(thk > 0.0)
    if ice.size < 4:
        return {}
    last = ice[-1]
    sel = np.arange(1, max(last - 1, 2))  # behind the front, without the inflow cell
    H_ref, u_ref = C.thickness(xf[sel]), C.velocity(xf[sel])
    dH, du = thk[sel] - H_ref, u[sel] - u_ref
    return {
        "H_rmse": float(np.sqrt(np.mean(dH**2))),
        "H_maxrel": float(np.max(np.abs(dH) / H_ref)),
        "u_rmse": float(np.sqrt(np.mean(du**2))),
        "u_maxrel": float(np.max(np.abs(du) / u_ref)),
        "H_r2": float(
            1.0 - np.sum(dH**2) / np.sum((thk[sel] - thk[sel].mean()) ** 2)
        ),
    }


def cost(run, ts):
    base = os.path.join(OUTPUTS, run)
    try:
        wall = float(open(os.path.join(base, ".t1")).read()) - float(
            open(os.path.join(base, ".t0")).read()
        )
    except OSError:
        wall = float(ts["wall"].iloc[-1])
    years = float(ts["t"].iloc[-1] - ts["t"].iloc[0])
    return {
        "wall_s": wall,
        "s_per_yr": wall / max(years, 1e-9),
        "steps": int(ts["step"].iloc[-1]),
    }


def metrics(run):
    ts = read_ts(run)
    res, dx = resolution(run)
    out = {"run": run, "res": res, "dx_km": dx / 1e3, "t_end": float(ts["t"].iloc[-1])}
    x = ts["x_front"].to_numpy()
    out["x_front_km"] = x[-1] / 1e3
    if not run.startswith("adv_none"):
        ref = front_reference(run, ts)
        err = x - ref
        out["front_rmse_km"] = float(np.sqrt(np.mean(err**2))) / 1e3
        out["front_maxerr_km"] = float(np.max(np.abs(err))) / 1e3
        out["x_ref_end_km"] = ref[-1] / 1e3
        out["front_err_end_km"] = (x[-1] - ref[-1]) / 1e3
    inflow = max(float(ts["inflow"].iloc[-1]), 1.0)
    out["mass_residual_rel"] = float(np.max(np.abs(ts["residual"]))) / inflow
    out.update(profile_errors(run, dx))
    out.update(cost(run, ts))
    return out


EXPERIMENTS = ("adv_", "retreat_", "advance_", "const_", "vm_")


def runs_with_output():
    """The campaign runs (prep/make_experiments.py) in outputs/ with an albrecht_ts.csv."""
    base = OUTPUTS
    if not os.path.isdir(base):
        return []
    return sorted(
        r
        for r in os.listdir(base)
        if r.startswith(EXPERIMENTS)
        and os.path.exists(os.path.join(base, r, "albrecht_ts.csv"))
    )


def markdown(df):
    """A pipe table of df (pandas' to_markdown needs tabulate, which igm does not install)."""
    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, row in df.iterrows():
        cells = [
            "" if pd.isna(v) else f"{v:.4g}" if isinstance(v, float) else str(v)
            for v in row
        ]
        lines.append("| " + " | ".join([str(name)] + cells) + " |")
    return "\n".join(lines)


def main(runs):
    rows = [metrics(r) for r in (runs or runs_with_output())]
    if not rows:
        print("no runs with albrecht_ts.csv in outputs/")
        return
    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 200)
    print(df.round(4).to_string())
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "summary.md"), "w") as f:
        f.write("# Albrecht flowline: summary (analysis.py)\n\n")
        f.write(markdown(df))
        f.write("\n")
    df.to_csv(os.path.join(ROOT, "results", "summary.csv"))


if __name__ == "__main__":
    main(sys.argv[1:])
