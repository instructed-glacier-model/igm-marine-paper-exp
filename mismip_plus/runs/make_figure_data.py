#!/usr/bin/env python3
"""Write the data of Fig. 3 (../figure/data/*.csv) from the run outputs and the MISMIP+ ensemble.

  igm_timeseries.csv      t_yr, xGL_centre_m, groundedArea_m2, iceVAF_m3 of the six Ice1 / Ice2 runs with the
                          neural-network mapping (outputs/<exp>_1km_nn/mismip_ts.csv)
  ensemble_envelope.csv   for Ice1 and Ice2, each branch (on: Ice{k}r then Ice{k}rr; off: Ice{k}ra), each
                          quantity (gl: grounding line on the centerline, m; area: change in grounded area, m2;
                          vaf: change in volume above flotation, m3) and each subset of Cornford et al. (2020)
                          (main: the 24 models of the main subset; weertman: its 11 Weertman models): the min and
                          max over the members every 10 yr, and the number of members n
  igm_newton_cg.csv       the same time series of the identity-mapping (Newton-CG) runs, for the comparison of
                          the text (written when outputs/<exp>_1km exist)

The ensemble is read from raw/ensemble/submission_data (prep/fetch_ensemble.py; Zenodo 10.5281/zenodo.3611936).
Usage: python make_figure_data.py [--outputs DIR]   (default: outputs/ of this folder)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ensemble as E  # noqa: E402

FIGURE_DATA = os.path.join(os.path.dirname(HERE), "figure", "data")
EXPS = ["Ice1r", "Ice1rr", "Ice1ra", "Ice2r", "Ice2rr", "Ice2ra"]
COLS = ["t", "xGL_centre", "groundedArea", "iceVAF"]
KEYS = {  # quantity: ensemble key, relative to t = 0 of Ice{k}r
    "gl": ("x_centre", False),
    "area": ("groundedArea", True),
    "vaf": ("iceVAF", True),
}


def igm_table(outputs, suffix):
    rows = []
    for exp in EXPS:
        path = os.path.join(outputs, f"{exp.lower()}_1km{suffix}", "mismip_ts.csv")
        if not os.path.exists(path):
            return None
        d = pd.read_csv(path)[COLS].rename(columns={"t": "t_yr", "xGL_centre": "xGL_centre_m",
                                                   "groundedArea": "groundedArea_m2", "iceVAF": "iceVAF_m3"})
        d.insert(0, "exp", exp)
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def ensemble_branch(subs, members, k, branch, key, rel):
    """(times, values[member, time]) of one branch: 'on' = Ice{k}r (0-100) then Ice{k}rr (100-200); 'off' =
    Ice{k}ra (100-200). Relative values are against Ice{k}r at t = 0."""
    if branch == "on":
        tt = np.arange(0.0, 201.0, 10.0)
        exps = [f"Ice{k}r" if t < 100.5 else f"Ice{k}rr" for t in tt]
    else:
        tt = np.arange(100.0, 201.0, 10.0)
        exps = [f"Ice{k}ra"] * tt.size
    vals = np.array([[E.branch_value(subs, s, e, key, t) for e, t in zip(exps, tt)] for s in members])
    if rel:
        v0 = np.array([E.value_at(subs[s].get(f"Ice{k}r"), key, 0.0) for s in members])
        vals = vals - v0[:, None]
    return tt, vals


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()
    os.makedirs(FIGURE_DATA, exist_ok=True)

    igm_table(args.outputs, "_nn").to_csv(os.path.join(FIGURE_DATA, "igm_timeseries.csv"), index=False)
    cg = igm_table(args.outputs, "")
    if cg is not None:
        cg.to_csv(os.path.join(FIGURE_DATA, "igm_newton_cg.csv"), index=False)

    subs = E.ensemble()
    sets = E.subsets(subs)
    rows = []
    for k in (1, 2):
        for branch in ("on", "off"):
            for q, (key, rel) in KEYS.items():
                for name in ("main", "weertman"):
                    tt, v = ensemble_branch(subs, sets[name], k, branch, key, rel)
                    with np.errstate(all="ignore"):
                        lo, hi = np.nanmin(v, 0), np.nanmax(v, 0)
                    rows += [dict(k=k, branch=branch, quantity=q, subset=name, n=len(sets[name]), t_yr=t, min=a, max=b)
                             for t, a, b in zip(tt, lo, hi)]
    pd.DataFrame(rows).to_csv(os.path.join(FIGURE_DATA, "ensemble_envelope.csv"), index=False, float_format="%.10g")
    print(f"written {os.path.relpath(FIGURE_DATA, os.getcwd())}/{{igm_timeseries,ensemble_envelope"
          f"{',igm_newton_cg' if cg is not None else ''}}}.csv  (main subset: {len(sets['main'])} models, "
          f"Weertman: {len(sets['weertman'])})")


if __name__ == "__main__":
    main()
