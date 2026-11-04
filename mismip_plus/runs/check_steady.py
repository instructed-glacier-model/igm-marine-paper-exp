"""
Drift of a run over its last WINDOW years, from outputs/RUN/mismip_ts.csv, against the
Ice0 steady-state criteria of this repo (README "Ice0 spin-up"):

  |x_GL(y = 0)| change  <= 0.5 km      over the window
  |VAF| relative change <= 1e-3        over the window

Also prints the wall-row grounding lines (both walls: a mirror-symmetry check), the volume
change and the recent rates.

Usage:  python check_steady.py RUN [RUN ...] [--window 1000]   (from this folder)
"""

import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
TOL_GL = 500.0  # m
TOL_VAF = 1.0e-3


def drift(run, window):
    ts = pd.read_csv(os.path.join(ROOT, "outputs", run, "mismip_ts.csv"))
    t1 = ts.t.iloc[-1]
    a = ts[ts.t >= t1 - window].iloc[0]
    b = ts.iloc[-1]
    d_gl = b.xGL_centre - a.xGL_centre
    d_vaf = (b.iceVAF - a.iceVAF) / b.iceVAF
    d_vol = (b.iceVolume - a.iceVolume) / b.iceVolume
    ok = abs(d_gl) <= TOL_GL and abs(d_vaf) <= TOL_VAF
    span = b.t - a.t
    print(f"{run}: t = {a.t:.0f} -> {b.t:.0f} yr ({len(ts)} rows)")
    print(f"  x_GL(y=0)   {a.xGL_centre / 1e3:8.2f} -> {b.xGL_centre / 1e3:8.2f} km  (change {d_gl / 1e3:+.3f} km)")
    print(f"  x_GL walls  S {b.xGL_south / 1e3:.2f}  N {b.xGL_north / 1e3:.2f} km")
    print(f"  VAF         {b.iceVAF / 1e9:10.1f} km3   (change {d_vaf:+.2e}, {d_vaf / span * 1e3:+.2e} per kyr)")
    print(f"  volume      {b.iceVolume / 1e9:10.1f} km3   (change {d_vol:+.2e})")
    print(f"  grounded    {b.groundedArea / 1e6:10.1f} km2;  max speed {b.umax:.0f} m/yr")
    print(f"  steady (|dGL| <= {TOL_GL / 1e3:g} km, |dVAF|/VAF <= {TOL_VAF:g} over {window:g} yr): "
          f"{'YES' if ok else 'NO'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=float, default=1000.0)
    args = ap.parse_args()
    for r in args.runs:
        drift(r, args.window)


if __name__ == "__main__":
    main()
