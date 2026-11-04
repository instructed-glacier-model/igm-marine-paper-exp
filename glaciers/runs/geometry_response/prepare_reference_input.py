#!/usr/bin/env python3
"""Input for a same-geometry reference solve of a transient run at YEAR.

The run's output is written after `thk` has updated the geometry, so the velocity
saved at YEAR belongs to the geometry the ice-flow solve saw, which the
iceflow_geometry user process saves as iceflow_input_thk / iceflow_input_usurf. The
reference input is the run's original input file (tau_ref, smb, water_level, obs,
...) with thk and usurf replaced by that geometry.

With --probe, the geometry is read from a cfl_probe dump (outputs/RUN/cfl_probe_STEP.npz, written
by the diagnostic user output cfl_probe) instead of an output.nc frame; YEAR is then only recorded.

Usage (from glaciers/runs): python geometry_response/prepare_reference_input.py outputs/RUN YEAR DESTINATION.nc [--probe FILE.npz]
"""

import argparse
import os

import numpy as np
import xarray as xr
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # glaciers/runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="outputs/RUN (with output.nc and .hydra/config.yaml)")
    ap.add_argument("year", type=float)
    ap.add_argument("destination")
    ap.add_argument("--probe", default=None, help="cfl_probe_STEP.npz to take the geometry from")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.run, ".hydra", "config.yaml")))
    source = os.path.join(ROOT, cfg["core"]["folder_data"], cfg["inputs"]["local"]["filename"])
    if args.probe:
        dump = np.load(args.probe)
        thk, usurf = dump["iceflow_input_thk"], dump["iceflow_input_usurf"]
        args.year = float(dump["t"])
    else:
        thk, usurf = frame_geometry(args.run, args.year)
    write(source, thk, usurf, args)


def frame_geometry(run, year):
    with xr.open_dataset(os.path.join(run, "output.nc")) as out:
        missing = [v for v in ("iceflow_input_thk", "iceflow_input_usurf") if v not in out]
        if missing:
            raise SystemExit(f"{run}/output.nc lacks {', '.join(missing)} "
                             "(add the iceflow_geometry process and save them)")
        t = np.asarray(out.time.values, dtype=float)
        k = int(np.argmin(np.abs(t - year)))
        if not np.isclose(t[k], year, atol=1e-4):
            raise SystemExit(f"year {year:g} not saved in {run}; nearest {t[k]:g}")
        return out.iceflow_input_thk.isel(time=k).values, out.iceflow_input_usurf.isel(time=k).values


def write(source, thk, usurf, args):
    ref = xr.open_dataset(source).load()
    ref["thk"] = (("y", "x"), thk.astype("float32"))
    ref["usurf"] = (("y", "x"), usurf.astype("float32"))
    ref["icemask"] = (("y", "x"), (thk > 0).astype("float32"))
    if "Href" in ref:  # a transient start file's partial cells belong to t = 0; the ice flow
        ref["Href"] = (("y", "x"), np.zeros_like(thk, dtype="float32"))  # sees thk only
    where = f"cfl_probe dump {os.path.basename(args.probe)}" if args.probe else "output.nc"
    ref.attrs.update(reference_geometry=f"iceflow input geometry of {args.run} at {args.year:g} ({where})",
                     reference_source=os.path.abspath(source))
    os.makedirs(os.path.dirname(os.path.abspath(args.destination)), exist_ok=True)
    ref.to_netcdf(args.destination)
    print(f"written {args.destination}")


if __name__ == "__main__":
    main()
