#!/usr/bin/env python3
"""Export the derived data of the GMD figures (tools/paper/fig_gmd_*.py) into the stand-alone hand-off package.

The package (default /home/tgregov/work/igm-32/gmd_marine_application) redraws the figures without the run folders
or this project; this script writes its data/ and configs/ from the runs, with the project's own readers (scoring,
transient_report, plot_trial_figure, fig_gmd_glaciers), so that every value is the one the figures use here:
  data/maps_GLACIER.nc       full grids: Newton-CG at t = 0 and at the end, the network run with N = 20 at the end
                             (thk, topg, water_level, velsurf_mag), the domain mask without PIG's quasi-rigid band
                             (keep), the flowline, the densities of both runs
  data/timeseries.csv        marine_ts of the 15 runs (transient_report.read_ts), long form
  data/errors.csv            the per-frame errors against Newton-CG of the 12 network runs (tools/vs_exact_report.py)
  data/costs.csv             per run: wall time, its parts (plot_trial_figure.cost_parts), node, igm code state,
                             init_stop's stop
  data/geometry_response.csv studies/error_decomposition/results/rosier_errors.csv
  configs/RUN.yaml           the resolved Hydra configuration of each run; configs/RUN_provenance.txt its
                             igm_version.txt (host, GPU, igm commit and diff, input and user-code hashes)
It also copies itself to scripts/export_data.py. Drawing needs none of this (scripts/make_figures.py).

Usage (from the project, CPU): ~/anaconda3/envs/igm-32/bin/python tools/paper/export_gmd_application.py [--out DIR]
"""

import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gmd_common as C  # noqa: E402
import fig_gmd_glaciers as G  # noqa: E402  (frame, flowline)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

PACKAGE = "/home/tgregov/work/igm-32/gmd_marine_application"
MAP_VARS = ["thk", "topg", "water_level", "velsurf_mag"]
CRS = {"jak": "EPSG:3413", "pig": "EPSG:3031", "men": "EPSG:32608 (UTM 8N)"}
NBIT_MAP = 20  # the network run of the difference maps
CLEAN_DIFF = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"  # sha256 of an empty diff


def run_dir(run):
    return os.path.join(C.OUT, run)


def all_runs(g):
    return [(None, g["exact"])] + list(g["runs"].items())


def export_maps(g, out):
    exact, net = g["exact"], g["runs"][NBIT_MAP]
    rho_x, rho_n = C.S.densities(run_dir(exact)), C.S.densities(run_dir(net))
    keep = C.S.domain(run_dir(exact))
    with xr.open_dataset(os.path.join(run_dir(exact), "output.nc")) as ds:
        x, y = ds.x.values, ds.y.values
        t0 = float(ds.time.values[0])
    states = {"exact_t0": (exact, t0), "exact_end": (exact, g["years"]), "net20_end": (net, g["years"])}
    data = {}
    for state, (run, t) in states.items():
        f = G.frame(run, t)
        if f is None:
            raise ValueError(f"{run}: no frame at {t:g} yr")
        for v in MAP_VARS:
            data[f"{state}_{v}"] = (("y", "x"), np.asarray(f[v].values).squeeze())
    if keep is None:
        keep = np.ones((len(y), len(x)), dtype=bool)
    data["keep"] = (("y", "x"), keep.astype("int8"))
    cfg_flow = G.yaml.safe_load(open(os.path.join(run_dir(exact), ".hydra", "config.yaml")))
    flow = np.loadtxt(
        os.path.join(C.ROOT, cfg_flow["outputs"]["marine_ts"]["flowline"]), delimiter=",", comments="#", skiprows=2
    )
    data["flowline_x"] = (("point",), flow[:, 1])
    data["flowline_y"] = (("point",), flow[:, 2])
    ds = xr.Dataset(data, coords={"x": ("x", x), "y": ("y", y)})
    ds.x.attrs.update(units="m")
    ds.y.attrs.update(units="m")
    ds.keep.attrs.update(
        long_name="1: inside the scored domain (0: PIG's quasi-rigid band, drawn as land and left out of every error)"
    )
    for state, (run, t) in states.items():
        for v in MAP_VARS:
            ds[f"{state}_{v}"].attrs.update(run=run, time_yr=t)
    ds.attrs.update(
        glacier=g["key"],
        name=g["name"],
        crs=CRS[g["key"]],
        dx=g["dx"],
        years=g["years"],
        exact_run=exact,
        network_run=net,
        network_nbit=NBIT_MAP,
        rho_i_exact=rho_x[0],
        rho_w_exact=rho_x[1],
        rho_i_network=rho_n[0],
        rho_w_network=rho_n[1],
        flowline_source=cfg_flow["outputs"]["marine_ts"]["flowline"],
        states="exact_t0: Newton-CG at t = 0; exact_end: Newton-CG at the end; net20_end: network N = 20 at the end",
    )
    path = os.path.join(out, "data", f"maps_{g['key']}.nc")
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print(f"written {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def export_timeseries(out):
    frames = []
    for g in C.GLACIERS:
        for n, run in all_runs(g):
            ts, _ = C.TR.read_ts(run)
            ts.insert(0, "N", n if n is not None else "")
            ts.insert(0, "run", run)
            ts.insert(0, "glacier", g["key"])
            frames.append(ts)
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "data", "timeseries.csv"), index=False)
    print(f"written data/timeseries.csv ({len(df)} rows)")


def export_errors(out):
    frames = []
    for g in C.GLACIERS:
        err = C.errors(g).copy()
        nbit = {run: n for n, run in g["runs"].items()}
        err.insert(1, "N", err.run.map(nbit))
        err.insert(0, "glacier", g["key"])
        frames.append(err.sort_values(["N", "t"]))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "data", "errors.csv"), index=False)
    print(f"written data/errors.csv ({len(df)} rows)")


def provenance(run):
    text = open(os.path.join(run_dir(run), "igm_version.txt")).read()

    def get(key):
        m = re.search(rf"^{key}: (.*)$", text, flags=re.M)
        return m.group(1).strip() if m else ""

    host = re.search(r"^host: (\S+)\s+CUDA_VISIBLE_DEVICES: (\S+)", text, flags=re.M)
    diff = get("igm_diff_sha256")
    return dict(
        experiment=get("experiment"),
        overrides=get("overrides"),
        node=host.group(1).split(".")[0] if host else "",
        gpu_index=host.group(2) if host else "",
        gpu=get("gpu"),
        igm_head=get("igm_head").split()[0][:8],
        igm_code="clean" if diff == CLEAN_DIFF else f"diff {diff[:8]}",
        igm_modified=get("igm_modified"),
    )


def init_stop(run):
    log = open(os.path.join(run_dir(run), "igm.log"), errors="ignore").read()
    m = re.search(r"Initial training (stopped|completed): ([^─╮]+?)\s*─", log)
    it = re.search(r"Iterations\s+([\d,]+) of ([\d,]+)", log)
    gr = re.search(r"Grounded\s+([\d.]+) %", log)
    fl = re.search(r"Floating\s+([\d.]+) %", log)
    return dict(
        init_stop_reason=m.group(2).strip() if m else "",
        init_stop_iterations=int(it.group(1).replace(",", "")) if it else np.nan,
        init_stop_error_grounded_pct=float(gr.group(1)) if gr else np.nan,
        init_stop_error_floating_pct=float(fl.group(1)) if fl else np.nan,
    )


def export_costs(out):
    rows = []
    for g in C.GLACIERS:
        for n, run in all_runs(g):
            parts = C.PT.cost_parts(run)
            row = dict(glacier=g["key"], run=run, N=n if n is not None else "", rc=C.rc(run))
            row.update(
                wall_min=C.PT.wall_minutes(run),
                startup_min=parts[0],
                direct_min=parts[1],
                training_min=parts[2],
                stepping_min=parts[3],
            )
            row.update(provenance(run))
            if n is not None:
                row.update(init_stop(run))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "data", "costs.csv"), index=False)
    print(f"written data/costs.csv ({len(df)} rows)")


def export_configs(out):
    for g in C.GLACIERS:
        for _, run in all_runs(g):
            shutil.copy(
                os.path.join(run_dir(run), ".hydra", "config.yaml"), os.path.join(out, "configs", f"{run}.yaml")
            )
            shutil.copy(
                os.path.join(run_dir(run), "igm_version.txt"), os.path.join(out, "configs", f"{run}_provenance.txt")
            )
    print("written configs/ (15 runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=PACKAGE)
    a = ap.parse_args()
    issues = C.data_status()
    if issues:
        raise SystemExit("incomplete data: " + "; ".join(issues))
    for sub in ("data", "configs", "scripts"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    for g in C.GLACIERS:
        export_maps(g, a.out)
    export_timeseries(a.out)
    export_errors(a.out)
    export_costs(a.out)
    shutil.copy(
        os.path.join(C.ROOT, "studies", "error_decomposition", "results", "rosier_errors.csv"),
        os.path.join(a.out, "data", "geometry_response.csv"),
    )
    print("written data/geometry_response.csv")
    export_configs(a.out)
    shutil.copy(os.path.abspath(__file__), os.path.join(a.out, "scripts", "export_data.py"))
    print("written scripts/export_data.py")


if __name__ == "__main__":
    main()
