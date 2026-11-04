#!/usr/bin/env python3
"""Error split of Sect. 5.5 of the paper (Fig. 8; after Rosier et al., egusphere-2026-5523, Sect. 3.3, Eq. 11), for
the neural-network runs with N = 20.

With E the frames of the network run, R those of the reference run (identity mapping solved with Newton-CG), and V
the mean surface speed over the mask A of the cells with more than 1 m of ice in both states (Pine Island's
quasi-rigid band left out):
  e_tot  = V_net(E) - V_ex(R)   the network run against the reference run
  e_inst = V_net(R) - V_ex(R)   instantaneous error: the network of time t evaluated on the reference state
  e_geom = V_net(E) - V_net(R)  geometry-induced: the same network on the two geometries (e_tot = e_inst + e_geom)
  e_phys = V_ex(E) - V_ex(R)    the response of the identity mapping (Newton-CG) to the same state difference
Sources, in outputs/ (see run.sh):
  V_net(E)  outputs/w_RUN                 the network run again, with its weights saved at every frame (bitwise the same)
  V_net(R)  outputs/netreplay_exact_RUN   the weights of each frame evaluated on the reference state, without training
  V_ex(E)   outputs/replay_RUN (+ _dbl)   cold Newton-CG solves of every frame of the network run
  V_ex(R)   outputs/replay_REFERENCE      cold Newton-CG solves of every frame of the reference run
e_phys leaves out the cells where V_ex(E) runs away (Pine Island's nearly detached floating block): cells where it
exceeds CAP times the largest V_ex(R) of the frame. The same split of the field (relative RMSE, divided by
||u_ex(R)||, columns R_*) is written too, with R_tot <= R_inst + R_geom.

Usage: python geometry_response.py [--outputs DIR] [--out CSV]
  --outputs  the folder of the run outputs (default: ../outputs)
  --out      the CSV to write (default: ../../figure/data/geometry_response.csv)
"""

import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.dirname(HERE)  # glaciers/runs
CAP = 2.0  # cells with u_ex(E) > CAP x the largest u_ex(R) of the frame are left out of e_phys
RUNS = {  # glacier: (reference run, network run with N = 20)
    "jak": ("glacier_jak_P1_30_exact_fast_mt5_25yr", "j25_n25_20_mt5"),
    "pig": ("glacier_pig_exact_fast_mt5_100yr", "pig_n50_20_mt5_100yr"),
    "men": ("glacier_men_exact_fast_mt5_25yr", "m25_n50_20_mt5"),
}


def frames(outputs, run):
    """[(dataset, times)] of a run; for a replay also its double-precision tail replay_RUN_dbl, if any (one PIG
    replay slowed down in single precision on the nearly detached block and was finished in double)."""
    out = []
    for name in (run, run + "_dbl") if run.startswith("replay_") else (run,):
        path = os.path.join(outputs, name, "output.nc")
        if os.path.exists(path):
            ds = xr.open_dataset(path)
            out.append((ds, np.asarray(ds.time.values, dtype=float)))
    if not out:
        raise SystemExit(f"missing {os.path.join(outputs, run, 'output.nc')}")
    return out


def at(parts, t):
    """The frame at time t, from the last part that holds it (the tail wins over the stopped replay)."""
    for ds, times in reversed(parts):
        k = int(np.argmin(np.abs(times - t)))
        if abs(times[k] - t) < 1e-4:
            return ds.isel(time=k)
    return None


def field(ds, name):
    return np.asarray(ds[name].values, dtype="float64").squeeze()


def geometry(ds):
    """Thickness the velocity of this frame belongs to."""
    return field(ds, "iceflow_input_thk" if "iceflow_input_thk" in ds else "thk")


def domain(outputs, run):
    """Ice mask of the run's input without the quasi-rigid band (Pine Island), or None."""
    cfg = yaml.safe_load(open(os.path.join(outputs, run, ".hydra", "config.yaml")))
    path = os.path.join(RUNS_DIR, cfg["core"]["folder_data"], cfg["inputs"]["local"]["filename"])
    with xr.open_dataset(path) as ds:
        if "bandmask" in ds:
            return ~(ds.bandmask.values > 0.5)
    return None


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def split(outputs, glacier, exact, run):
    keep = domain(outputs, exact)
    net, nr = frames(outputs, f"w_{run}"), frames(outputs, f"netreplay_exact_{run}")
    xe, xr_ = frames(outputs, f"replay_{run}"), frames(outputs, f"replay_{exact}")
    ext = frames(outputs, exact)
    rows = []
    for t in net[0][1]:
        fE, fNR, fXE, fXR, fx = at(net, t), at(nr, t), at(xe, t), at(xr_, t), at(ext, t)
        if any(f is None for f in (fE, fNR, fXE, fXR, fx)):
            continue
        a = (geometry(fE) > 1.0) & (geometry(fx) > 1.0)
        if keep is not None:
            a &= keep
        uE, uNR, uXE, uXR = (field(f, "velsurf_mag") for f in (fE, fNR, fXE, fXR))
        a &= np.isfinite(uE) & np.isfinite(uNR) & np.isfinite(uXR)
        if a.sum() < 10:
            continue
        p = a & np.isfinite(uXE)
        wild = p & (uXE > CAP * float(np.max(uXR[a])))
        p &= ~wild
        stalled = float(np.max(uXE[p])) < 1e-3
        V = lambda u, m: float(np.mean(u[m]))  # noqa: E731
        scale = np.linalg.norm(uXR[a])
        rows.append({
            "glacier": glacier, "run": run, "t": float(t), "cells": int(a.sum()), "excluded_phys": int(wild.sum()),
            "V_exR": V(uXR, a),
            "e_tot": V(uE, a) - V(uXR, a), "e_inst": V(uNR, a) - V(uXR, a), "e_geom": V(uE, a) - V(uNR, a),
            "e_phys": np.nan if stalled else V(uXE, p) - V(uXR, p),
            "R_tot": 100 * np.linalg.norm((uE - uXR)[a]) / scale,
            "R_inst": 100 * np.linalg.norm((uNR - uXR)[a]) / scale,
            "R_geom": 100 * np.linalg.norm((uE - uNR)[a]) / scale,
            "R_phys": np.nan if stalled else 100 * np.linalg.norm((uXE - uXR)[p]) / np.linalg.norm(uXR[p]),
            "cos_inst_geom": cosine((uNR - uXR)[a], (uE - uNR)[a]),  # < 0: the two error fields cancel
            "cos_geom_phys": np.nan if stalled else cosine((uE - uNR)[p], (uXE - uXR)[p]),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", default=os.path.join(RUNS_DIR, "outputs"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(RUNS_DIR), "figure", "data", "geometry_response.csv"))
    args = ap.parse_args()
    tables = []
    for glacier, (exact, run) in RUNS.items():
        tables.append(split(args.outputs, glacier, exact, run))
        print(f"{glacier}: {len(tables[-1])} frames")
    err = pd.concat(tables, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    err.to_csv(args.out, index=False)
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
