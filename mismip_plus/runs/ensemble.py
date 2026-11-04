"""
The MISMIP+ ensemble (Cornford et al., 2020; prep/fetch_ensemble.py -> raw/ensemble/) and
IGM results in the same submission format (results/, tools/to_mismip.py), read the same way.

Submission files differ in their conventions; each file is normalised here:
  * variable names: iceVAF / iceVaf, time / nTime / ntime (case-insensitive lookup);
  * dimensions: the xGL/yGL time dimension is the one of the length of time;
  * units: coordinates in km (PSU3D) are converted to m (max |y| < 1000 -> km);
  * fill values (9.97e36) are masked, times after 200 yr (extended rr/ra runs) dropped;
  * duplicated times (some BISICLES files hold appended restart segments) keep the first
    record in file order: in Ice1rr_SCO_BISICLES_SSA_Schoof_250m the second segment
    restarts its time counter at 100 yr but continues the run beyond 300 yr;
  * half-domain submissions (WAVI: y in [0, 40] km) are used as they are: the centre line
    and one wall are inside them.
y is the submission coordinate, y in [0, 80] km (centre line at 40 km).

Grounding-line positions from the xGL/yGL points, per time:
  x_centre  median x of the points nearest to the centre line (within 500 m of the
            nearest point's distance; coarse models have no point on the centre line);
  x_wall    median x of the points within 1 km of a side wall (nan if none).

Metadata: friction law and stress approximation from the submission name, which follows
Cornford et al. (2020, Table 2); DNG_STREAMICE (Schoof, L1Lx) is set by hand. Left out:
the WAVI "pgoff" files (the partial Ice2r supplement, Cornford et al., 2020, Sect. 3.11)
and HGU_UA_SSA (its two files are byte-identical to HGU_UA_SSA_Weertman).

Subsets, following Cornford et al. (2020, Sect. 4.2):
  all        every submission (Cornford's "all models")
  main       SSA, L1Lx and HO models (not HySSA or full Stokes) that completed all seven
             experiments, with Ice0 x_centre(t = 0) in 450 +- 15 km and Ice1r
             x_centre(100 yr) in 385 +- 35 km
  weertman   main models with the Weertman law; coulomb: main with Schoof or Tsai
  ho         every Blatter-Pattyn (higher-order) submission, the class of IGM's MOLHO
  ho_weertman  the HO + Weertman ones (ISSM 1 km, MALI 500 m): IGM's closest analogues
"""

import glob
import os
import re

import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE = os.path.join(ROOT, "raw", "ensemble", "submission_data")
RESULTS = os.path.join(ROOT, "results")
EXPERIMENTS = ["Ice0", "Ice1r", "Ice1ra", "Ice1rr", "Ice2r", "Ice2ra", "Ice2rr"]
Y_CENTRE = 40.0e3
T_MAX = 200.0 + 1e-6
MANUAL = {"DNG_STREAMICE": {"friction": "Schoof", "stress": "L1Lx", "dx": None}}
SKIP = ("HGU_UA_SSA",)


def meta(sub):
    """Friction law, stress class and resolution (m) of a submission name."""
    if sub in MANUAL:
        return dict(MANUAL[sub])
    tok = sub.split("_")
    friction = next((t for t in tok if t in ("Weertman", "Schoof", "Tsai")), "unknown")
    if "SSA+SIA" in tok or any(t.startswith("L1L2") for t in tok):
        stress = "L1Lx"
    elif "HySSA" in tok:
        stress = "HySSA"
    else:
        stress = next((t for t in tok if t in ("SSA", "HO", "FS")), "unknown")
    dx = None
    m = re.search(r"_(\d+)(km|m)$", sub)
    if m:
        dx = float(m.group(1)) * (1e3 if m.group(2) == "km" else 1.0)
    return {"friction": friction, "stress": stress, "dx": dx}


def _var(ds, name):
    low = {v.lower(): v for v in ds.variables}
    return ds[low[name.lower()]] if name.lower() in low else None


def _masked(a):
    a = np.asarray(a, dtype=np.float64)
    return np.where(np.isfinite(a) & (np.abs(a) < 1e30), a, np.nan)


def read(path):
    """dict(time, iceVolume, iceVAF, groundedArea, x_centre, x_wall, xgl, ygl) of one file;
    xgl/ygl are lists (one array of points per time)."""
    with xr.open_dataset(path, mask_and_scale=True, decode_times=False) as ds:
        t = _masked(_var(ds, "time").values)
        nt = t.size
        X, Y = _var(ds, "xGL"), _var(ds, "yGL")  # (point, time) or (time, point)
        axis = [i for i, n in enumerate(X.shape) if n == nt]
        axis = axis[-1] if axis else X.ndim - 1
        X = np.moveaxis(_masked(X.values), axis, -1)
        Y = np.moveaxis(_masked(Y.values), axis, -1)
        scal = {k: (_masked(_var(ds, k).values) if _var(ds, k) is not None else np.full(nt, np.nan))
                for k in ("iceVolume", "iceVAF", "groundedArea")}
    if np.nanmax(np.abs(Y)) < 1000.0:  # km
        X, Y = X * 1e3, Y * 1e3
    keep = np.flatnonzero(np.isfinite(t) & (t <= T_MAX))
    _, first = np.unique(t[keep], return_index=True)  # sorted times, first occurrence
    idx = keep[first]
    out = {"time": t[idx], **{k: v[idx] for k, v in scal.items()}, "xgl": [], "ygl": []}
    xc, xw = [], []
    for k in idx:
        m = np.isfinite(X[:, k]) & np.isfinite(Y[:, k])
        x, y = X[m, k], Y[m, k]
        out["xgl"].append(x)
        out["ygl"].append(y)
        xc.append(x_centre(x, y))
        xw.append(x_wall(x, y))
    out["x_centre"], out["x_wall"] = np.array(xc), np.array(xw)
    return out


def x_centre(x, y):
    if x.size == 0:
        return np.nan
    d = np.abs(y - Y_CENTRE)
    return float(np.median(x[d <= d.min() + 500.0]))


def x_wall(x, y):
    sel = (y <= 1.0e3) | (y >= 79.0e3)
    return float(np.median(x[sel])) if sel.any() else np.nan


def ensemble():
    """{submission: {experiment: record}} for the whole ensemble, with 'meta'."""
    subs = {}
    for f in sorted(glob.glob(os.path.join(ENSEMBLE, "Ice*.nc"))):
        exp, sub = re.match(r"(Ice[0-9][a-z]*)_(.+)\.nc$", os.path.basename(f)).groups()
        if "_pgoff_" in sub or sub in SKIP:
            continue
        subs.setdefault(sub, {"meta": meta(sub)})[exp] = read(f)
    return subs


def value_at(rec, key, t):
    """Value of rec[key] at time t (nearest frame within 0.5 yr), nan otherwise."""
    if rec is None or rec["time"].size == 0:
        return np.nan
    k = int(np.argmin(np.abs(rec["time"] - t)))
    return float(rec[key][k]) if abs(rec["time"][k] - t) <= 0.5 else np.nan


def subsets(subs):
    """{name: [submissions]} (see the module docstring)."""
    main = []
    for s, v in subs.items():
        if v["meta"]["stress"] not in ("SSA", "L1Lx", "HO") or not all(e in v for e in EXPERIMENTS):
            continue
        g0 = value_at(v["Ice0"], "x_centre", 0.0)
        g1 = value_at(v["Ice1r"], "x_centre", 100.0)
        if abs(g0 - 450e3) <= 15e3 and abs(g1 - 385e3) <= 35e3:
            main.append(s)
    ho = [s for s in subs if subs[s]["meta"]["stress"] == "HO"]
    return {
        "all": list(subs),
        "main": main,
        "weertman": [s for s in main if subs[s]["meta"]["friction"] == "Weertman"],
        "coulomb": [s for s in main if subs[s]["meta"]["friction"] in ("Schoof", "Tsai")],
        "ho": ho,
        "ho_weertman": [s for s in ho if subs[s]["meta"]["friction"] == "Weertman"],
    }


def branch_value(subs, s, exp, key, t):
    """value_at, with the ra / rr branches taking the parent r run's value at their start
    (100 yr; some submissions have no frame there)."""
    v = value_at(subs[s].get(exp), key, t)
    if np.isnan(v) and exp.endswith(("ra", "rr")) and abs(t - 100.0) < 0.5:
        v = value_at(subs[s].get(exp[:-1]), key, t)
    return v


def envelope(subs, members, exp, key, times):
    """(mean, min, max, n) of rec[key] over members at the given times (nan where none)."""
    vals = np.array([[branch_value(subs, s, exp, key, t) for t in times] for s in members]).reshape(len(members), len(times))
    n = np.sum(np.isfinite(vals), axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(vals, 0), np.nanmin(vals, 0), np.nanmax(vals, 0), n


def igm(label):
    """{experiment: record} of an IGM configuration in results/ (tools/to_mismip.py)."""
    out = {}
    for exp in EXPERIMENTS:
        f = os.path.join(RESULTS, f"{exp}_{label}.nc")
        if os.path.exists(f):
            out[exp] = read(f)
    return out


if __name__ == "__main__":
    subs = ensemble()
    for name, members in subsets(subs).items():
        print(f"{name:9s} {len(members):2d}: {', '.join(members)}")
    print()
    print(f"{'submission':42s} {'fric':8s} {'stress':6s} {'Ice0 x_c':>9s} {'Ice1r 100':>9s} {'Ice1ra 200':>10s} {'Ice1rr 200':>10s} {'Ice2r 100':>9s}")
    for s, v in subs.items():
        m = v["meta"]
        row = [value_at(v.get("Ice0"), "x_centre", 0), value_at(v.get("Ice1r"), "x_centre", 100),
               value_at(v.get("Ice1ra"), "x_centre", 200), value_at(v.get("Ice1rr"), "x_centre", 200),
               value_at(v.get("Ice2r"), "x_centre", 100)]
        print(f"{s:42s} {m['friction']:8s} {m['stress']:6s} " + " ".join(f"{r / 1e3:9.1f}" for r in row))
