"""
Write the MISMIP+ input files data/mismip_plus_<res>.nc (igm `local` input).

Each file holds the bed, the initial state of the Ice0 spin-up, the sea level and the Ice2
melt field (100 m/yr for x > 480 km; igm bmb keeps it on the ocean-connected floating ice
only). The initial state is the "simple initial state" the protocol leaves free: a parabolic
ice-sheet profile along the centre line,
    h_c(x) = max(H0 sqrt(1 - x / L), h_min),   H0 2000 m, L 700 km, h_min 100 m,
whose SURFACE is flat across flow where the centre line is grounded (thickness = surface -
bed over the side walls), and whose thickness is h_c(x) where the centre line floats.
A thickness uniform across flow instead makes the surface copy the 500-m side walls of the
bed (lateral slopes ~0.1): the first solve then has ~30 km/yr lateral flow near the divide,
a violent transient that the emulator cannot follow (outputs/superseded/*_oldprofile).
A thin uniform slab (--slab) barely moves.

Usage:  python prep/make_inputs.py [--res 2km 1km 500m 250m] [--slab THK]
        (--slab writes data/mismip_plus_<res>_slab<THK>.nc)
"""

import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


H0, L0, H_MIN = 2000.0, 700.0e3, 100.0


def initial_profile(x):
    """Initial centre-line thickness along flow (m): a parabolic profile, floored at H_MIN."""
    return np.maximum(H0 * np.sqrt(np.clip(1.0 - np.asarray(x) / L0, 0.0, None)), H_MIN)


def initial_state(x, y):
    """Initial thickness (ny, nx): flat surface across flow where the centre line is grounded."""
    hc = initial_profile(x)
    bc = C.bed(x, [0.0])[0]
    lc, sc = C.surfaces(bc, hc)
    grounded_c = lc <= bc + 1e-9  # centre line resting on the bed
    b = C.bed(x, y)
    thk = np.where(grounded_c[None, :], np.maximum(sc[None, :] - b, H_MIN), hc[None, :])
    return thk


def dataset(x, y, thk, attrs=None):
    """igm input dataset on the MISMIP+ grid for the thickness thk (ny, nx)."""
    topg = C.bed(x, y)
    lsurf, usurf = C.surfaces(topg, thk)
    dims = ("y", "x")
    ds = xr.Dataset(
        {
            "topg": (dims, topg.astype("float32"), {"units": "m", "long_name": "bed elevation"}),
            "thk": (dims, thk.astype("float32"), {"units": "m", "long_name": "ice thickness"}),
            "lsurf": (dims, lsurf.astype("float32"), {"units": "m"}),
            "usurf": (dims, usurf.astype("float32"), {"units": "m"}),
            "water_level": (dims, np.full_like(topg, C.SEA_LEVEL, dtype="float32"), {"units": "m"}),
            "prescribed_melt_rate": (
                dims,
                C.ice2_melt(x, y).astype("float32"),
                {"units": "m ice eq. yr-1", "long_name": "MISMIP+ Ice2 melt before the shelf mask"},
            ),
        },
        coords={"x": ("x", x.astype("float32")), "y": ("y", y.astype("float32"))},
    )
    ds.attrs.update(
        {
            "title": "MISMIP+ (Asay-Davis et al., 2016; Cornford et al., 2020)",
            "rho_ice": C.RHO_ICE,
            "rho_water": C.RHO_WATER,
            "B6": C.B6,
            "dx": float(x[1] - x[0]),
        }
    )
    ds.attrs.update(attrs or {})
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", nargs="+", default=list(C.RESOLUTIONS))
    ap.add_argument("--slab", type=float, default=None, help="uniform initial thickness (m) instead of the profile")
    args = ap.parse_args()
    for res in args.res:
        x, y = C.grid(res)
        if args.slab is None:
            thk = initial_state(x, y)
            how = f"profile H0 {H0:g} m, L {L0 / 1e3:g} km, h_min {H_MIN:g} m, flat surface across flow"
        else:
            thk = np.full((y.size, x.size), args.slab)
            how = f"slab {args.slab:g} m"
        ds = dataset(x, y, thk, {"history": f"prep/make_inputs.py ({how})", "initial_state": how})
        tag = "" if args.slab is None else f"_slab{args.slab:g}"
        out = os.path.join(C.ROOT, "data", f"mismip_plus_{res}{tag}.nc")
        ds.to_netcdf(out)
        print(f"written {os.path.relpath(out, C.ROOT)}: {x.size} x {y.size} nodes at {x[1] - x[0]:g} m")


if __name__ == "__main__":
    main()
