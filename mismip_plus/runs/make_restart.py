"""
Restart file data/restart/OUT.nc from the frame at time T of outputs/RUN/output.nc.

The file is a normal igm `local` input (prep/make_inputs.py dataset): the thickness of the
frame, the analytic bed, sea level, the Ice2 melt field and the surfaces recomputed from
flotation. With --to-res the thickness is interpolated bilinearly onto the grid of that
resolution (grid sequencing of the Ice0 spin-up; both grids span the same domain).

Usage:  python make_restart.py RUN --out OUT [--time T] [--to-res 1km]   (from this folder)
        (T defaults to the last frame)
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "prep"))
import common as C  # noqa: E402
from make_inputs import dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--to-res", default=None, choices=list(C.RESOLUTIONS))
    args = ap.parse_args()
    src = os.path.join(ROOT, "outputs", args.run, "output.nc")
    with xr.open_dataset(src) as ds:
        times = ds.time.values.astype(float)
        k = len(times) - 1 if args.time is None else int(np.argmin(np.abs(times - args.time)))
        if args.time is not None and abs(times[k] - args.time) > 1e-3:
            raise SystemExit(f"no frame at t = {args.time} in {src} (frames: {times.tolist()})")
        thk = ds.thk.isel(time=k).values.astype(np.float64)
        x, y = ds.x.values.astype(np.float64), ds.y.values.astype(np.float64)
    if args.to_res:
        xn, yn = C.grid(args.to_res)
        f = RegularGridInterpolator((y, x), thk, method="linear")
        Y, X = np.meshgrid(yn, xn, indexing="ij")
        thk = np.maximum(f(np.stack([Y.ravel(), X.ravel()], axis=-1)).reshape(Y.shape), 0.0)
        x, y = xn, yn
    else:
        x, y = C.grid(float(x[1] - x[0]))  # exact float64 nodes
    how = f" regridded to {args.to_res}" if args.to_res else ""
    ds = dataset(x, y, thk, {"history": f"make_restart.py {args.run} t={times[k]:g}{how}",
                             "source_run": args.run, "source_time": float(times[k])})
    out = os.path.join(ROOT, "data", "restart", f"{args.out}.nc")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.to_netcdf(out)
    print(f"written {os.path.relpath(out, ROOT)}: {args.run} at t = {times[k]:g}{how}, {x.size} x {y.size}")


if __name__ == "__main__":
    main()
