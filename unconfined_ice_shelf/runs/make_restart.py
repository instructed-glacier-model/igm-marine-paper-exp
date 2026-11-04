"""
Last frame (or the frame at --time) of a run -> data/restart/<run>.nc, an igm `local` input.

It keeps the geometry and the front state (thk, Href, psi) and sets the edge column back to
H0: the igm Dirichlet inflow thickness is the initial thickness of that column, and a saved
frame holds the evolved one.

Usage:  python make_restart.py RUN [--time T]   (from this folder)
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "prep"))
import common as C  # noqa: E402

KEEP = ("topg", "thk", "Href", "psi")


def restart(run, time=None):
    with xr.open_dataset(os.path.join(ROOT, "outputs", run, "output.nc")) as ds:
        frame = ds.isel(time=-1) if time is None else ds.sel(time=time, method="nearest")
        out = xr.Dataset({k: frame[k].astype("float32") for k in KEEP if k in frame})
        out["thk"][:, 0] = C.H0
        out["water_level"] = xr.zeros_like(out["topg"])
        out.attrs["source"] = f"outputs/{run}/output.nc at t = {float(frame.time):g} yr"
        out = out.load()
    path = os.path.join(ROOT, "data", "restart", f"{run}.nc")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out.to_netcdf(path)
    print(f"wrote {path} ({out.attrs['source']})")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--time", type=float, default=None)
    args = parser.parse_args()
    restart(args.run, args.time)
