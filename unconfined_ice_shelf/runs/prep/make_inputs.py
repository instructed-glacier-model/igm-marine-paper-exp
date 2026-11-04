"""
Write the igm input files data/albrecht_<res>.nc (igm `local` input).

Each file holds the deep bed, the zero sea level, and the initial shelf: the analytic steady
profile over the first two cells (x from the inflow face), with the first column at H0 so
that the Dirichlet inflow thickness (igm thk boundary `dirichlet`: the initial thickness of
the edge column) is H0. The advancing front then starts at x = 2 dx, i.e. at the time
front_time(2 dx) of the analytic transient.

Usage:  python prep/make_inputs.py [--res 10km 5km 2.5km 1.25km]
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

INITIAL_CELLS = 2  # full cells of the initial shelf


def initial_thickness(x, dx, cells=INITIAL_CELLS):
    """Initial thickness on the nodes x (m): analytic at the cell centres, H0 in column 0."""
    face = np.asarray(x) + 0.5 * dx  # centre of each cell, from the inflow face
    thk = np.where(np.arange(len(x)) < cells, C.thickness(face), 0.0)
    thk[0] = C.H0
    return thk


def dataset(dx, thk_row):
    x, y = C.grid(dx)
    shape = (len(y), len(x))
    dims = ("y", "x")
    thk = np.broadcast_to(thk_row, shape).astype("float32")
    return xr.Dataset(
        {
            "topg": (dims, np.full(shape, C.BED, "float32"), {"units": "m"}),
            "thk": (dims, thk, {"units": "m"}),
            "water_level": (dims, np.zeros(shape, "float32"), {"units": "m"}),
        },
        coords={"x": ("x", x.astype("float32")), "y": ("y", y.astype("float32"))},
        attrs={"title": "Albrecht et al. (2011) unconfined ice-shelf flowline", "dx": dx},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", nargs="*", default=list(C.RESOLUTIONS))
    args = parser.parse_args()
    os.makedirs(os.path.join(C.ROOT, "data"), exist_ok=True)
    for res in args.res:
        dx = C.RESOLUTIONS[res]
        x, _ = C.grid(dx)
        path = os.path.join(C.ROOT, "data", f"albrecht_{res}.nc")
        dataset(dx, initial_thickness(x, dx)).to_netcdf(path)
        print(f"wrote {path} ({len(x)} x {C.NY} nodes)")


if __name__ == "__main__":
    main()
