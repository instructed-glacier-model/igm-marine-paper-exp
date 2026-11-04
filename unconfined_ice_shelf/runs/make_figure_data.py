#!/usr/bin/env python3
"""Write the data of Fig. 2 (../figure/data/*.csv) from the run outputs and the analytic solutions (prep/common.py).

  profiles_2.5km.csv   x_km, thickness_m, velocity_m_yr of the ice on the last frame of adv_<scheme>_2.5km (middle
                       row), and the analytic profiles (scheme "analytic") up to the analytic steady front
  steady_front.csv     dx_km, scheme, x_front_km: the final front of adv_<scheme>_<res>; analytic_km = x(H = 250 m)
  trajectories.csv     run, scheme, t_yr (from the start of the run), x_front_km, reference_km: the front of the
                       transient runs and its analytic reference (Eq. 13 for the advance; x0 +/- W t for the
                       prescribed velocities; dx/dt = u(x) - c on the analytic profile for von Mises calving)

Usage: python make_figure_data.py [--outputs DIR]   (default: outputs/ of this folder)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "prep"))
sys.path.insert(0, HERE)
import analysis  # noqa: E402
import common as C  # noqa: E402

FIGURE_DATA = os.path.join(os.path.dirname(HERE), "figure", "data")
SCHEMES = ("sub_grid", "level_set")
X_FRONT = C.position(C.H_CALVING)  # analytic steady front (m), 144.45 km


def profile(outputs, run):
    """x (km), H (m), u (m/yr) of the ice on the middle row of the last frame."""
    with xr.open_dataset(os.path.join(outputs, run, "output.nc")) as ds:
        frame = ds.isel(time=-1)
        mid = frame.sizes["y"] // 2
        thk = frame["thk"].values[mid]
        u = frame["ubar"].values[mid]
        x = frame["x"].values / 1e3
    ice = thk > 0.0
    return x[ice], thk[ice], u[ice]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()
    analysis.OUTPUTS = args.outputs
    os.makedirs(FIGURE_DATA, exist_ok=True)

    rows = []
    for scheme in SCHEMES:
        x, thk, u = profile(args.outputs, f"adv_{scheme}_2.5km")
        rows += [dict(scheme=scheme, x_km=a, thickness_m=b, velocity_m_yr=c) for a, b, c in zip(x, thk, u)]
    xa = np.linspace(0.0, X_FRONT, 400)
    rows += [dict(scheme="analytic", x_km=a / 1e3, thickness_m=b, velocity_m_yr=c)
             for a, b, c in zip(xa, C.thickness(xa), C.velocity(xa))]
    pd.DataFrame(rows).to_csv(os.path.join(FIGURE_DATA, "profiles_2.5km.csv"), index=False, float_format="%.6g")

    rows = []
    for scheme in SCHEMES:
        for res, dx in (("10km", 10.0), ("5km", 5.0), ("2.5km", 2.5)):
            ts = analysis.read_ts(f"adv_{scheme}_{res}")
            rows.append(dict(dx_km=dx, scheme=scheme, x_front_km=ts["x_front"].iloc[-1] / 1e3, analytic_km=X_FRONT / 1e3))
    pd.DataFrame(rows).to_csv(os.path.join(FIGURE_DATA, "steady_front.csv"), index=False, float_format="%.6g")

    rows = []
    for kind in ("adv", "advance", "retreat", "vm"):
        for scheme in SCHEMES:
            run = f"{kind}_{scheme}_5km"
            ts = analysis.read_ts(run)
            t = ts["t"].to_numpy() - ts["t"].iloc[0]
            ref = analysis.front_reference(run, ts) / 1e3
            rows += [dict(run=run, scheme=scheme, t_yr=a, x_front_km=b, reference_km=c)
                     for a, b, c in zip(t, ts["x_front"].to_numpy() / 1e3, ref)]
    pd.DataFrame(rows).to_csv(os.path.join(FIGURE_DATA, "trajectories.csv"), index=False, float_format="%.8g")
    print(f"written {os.path.relpath(FIGURE_DATA, os.getcwd())}/{{profiles_2.5km,steady_front,trajectories}}.csv")


if __name__ == "__main__":
    main()
