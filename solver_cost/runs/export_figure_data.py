#!/usr/bin/env python3
"""Export the data of the solver-scaling figure and of the probe-batching comparison (provenance script; it reads
the study outputs on the octopus cluster and is not needed to redraw the figure).

Writes, next to this script's parent folder:
  data/solver_scaling.csv   the 16 cases of the figure (one row per basis, precision and grid);
  data/probe_batch.csv      MOLHO cold solves at probe_batch 0, 1, 3 and 9 (1 km and 500 m);
  data/paper_case_grids.csv grid sizes and solver settings of the paper's test cases.
"""

import glob
import json
import os

import netCDF4
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = "/home/tgregov/work/igm-32"
STUDY = f"{ROOT}/igm-marine/studies/solver_scaling"
GIB = 1024.0


def scaling():
    d = pd.read_csv(f"{STUDY}/scaling.csv")
    d = d[d.variant == "main"].copy()
    oom = d.error.astype(str).str.contains("ResourceExhausted")
    out = pd.DataFrame(
        {
            "grid": d.res,
            "spacing_m": d.spacing_m,
            "basis": d.basis.str.upper(),
            "precision": d.precision,
            "nx": d.nx,
            "ny": d.ny,
            "dofs": d.dofs_raster,
            "dofs_active_ice": d.dofs_active_ice,
            "preconditioner": d.preconditioner,
            "probe_batch": d.probe_batch,
            "status": np.where(oom, "out_of_memory", "ok"),
            "cold_median_s": d.cold_median_s,
            "cold_min_s": d.cold_min_s,
            "cold_max_s": d.cold_max_s,
            "compile_call_s": np.where(oom, np.nan, d.compile_call_s),
            "newton_its": d.newton_its,
            "cg_its_per_newton": d.cg_per_newton,
            "time_per_newton_s": d.time_per_newton_s,
            "peak_gpu_gib": np.where(oom, np.nan, d.tf_peak_mib / GIB),
            "rel_rmse_vs_float64_ref": d.relrmse_all,
        }
    )
    out["plotted"] = out.cold_median_s.notna() & (out.rel_rmse_vs_float64_ref <= 0.01)
    order = {"4km": 0, "2km": 1, "1km": 2, "500m": 3}
    out = out.assign(_o=out.grid.map(order)).sort_values(["basis", "precision", "_o"])
    return out.drop(columns="_o")


def probe_batch():
    rows = []
    cases = sorted(glob.glob(f"{STUDY}/cases/probe_*_molho_*"))
    cases += [f"{STUDY}/work/prefix_058558bc/cases/1km_molho_f64"]
    for case in cases:
        name = os.path.basename(case)
        parts = name.replace("probe_", "").split("_")
        grid, prec = parts[0], {"f32": "float32", "f64": "float64"}[parts[2]]
        pb = int(parts[3][2:]) if len(parts) > 3 else 9
        reps = pd.read_csv(f"{case}/repetitions_raw.csv")
        comp = reps[reps.phase == "compile"]
        cold = reps[(reps.phase == "cold") & (reps.status == "ok")]
        oom = reps.error.astype(str).str.contains("ResourceExhausted").any()
        rows.append(
            {
                "grid": grid,
                "precision": prec,
                "probe_batch": pb,
                "status": "out_of_memory" if oom else "ok",
                "n_cold": len(cold),
                "compile_call_s": np.nan if oom else float(comp.wall_s.iloc[0]),
                "cold_median_s": cold.wall_s.median() if len(cold) else np.nan,
                "newton_its": cold.newton_its.median() if len(cold) else np.nan,
                "cg_its_total": cold.cg_its_total.median() if len(cold) else np.nan,
                "time_per_newton_s": (
                    (cold.wall_s / cold.newton_its).median() if len(cold) else np.nan
                ),
                "peak_gpu_gib": reps.tf_peak_mib.max() / GIB if not oom else np.nan,
                "source": name if "prefix" not in case else "1km_molho_f64 (main matrix)",
            }
        )
    out = pd.DataFrame(rows)
    out = out.assign(_g=out.grid.map({"1km": 0, "500m": 1}))
    return out.sort_values(["_g", "precision", "probe_batch"]).drop(columns="_g")


def grid_of(path):
    d = netCDF4.Dataset(path)
    x, y = d["x"][:], d["y"][:]
    return len(x), len(y), float(abs(x[1] - x[0]))


def paper_cases():
    cases = [
        ("Albrecht flowline (verification)", f"{ROOT}/igm-albrecht/data/albrecht_10km.nc", "MOLHO", "Newton-CG runs"),
        ("Albrecht flowline (verification)", f"{ROOT}/igm-albrecht/data/albrecht_5km.nc", "MOLHO", "Newton-CG runs"),
        ("Albrecht flowline (verification)", f"{ROOT}/igm-albrecht/data/albrecht_2.5km.nc", "MOLHO", "Newton-CG runs"),
        ("MISMIP+", f"{ROOT}/igm-mismip-plus/data/mismip_plus_1km.nc", "MOLHO", "emulator runs; Newton-CG check"),
        ("Mendenhall Glacier", f"{ROOT}/gmd_marine_application/data/maps_men.nc", "MOLHO", "Newton-CG reference"),
        ("Pine Island Glacier", f"{ROOT}/gmd_marine_application/data/maps_pig.nc", "MOLHO", "Newton-CG reference"),
        ("Sermeq Kujalleq", f"{ROOT}/gmd_marine_application/data/maps_jak.nc", "MOLHO", "Newton-CG reference"),
    ]
    rows = []
    for name, path, basis, role in cases:
        nx, ny, dx = grid_of(path)
        rows.append(
            dict(case=name, spacing_m=dx, nx=nx, ny=ny, nodes=nx * ny, basis=basis, probe_batch=9, newton_cg_role=role)
        )
    for grid in ("4km", "2km", "1km", "500m"):
        nx, ny, dx = grid_of(f"{ROOT}/igm-marine/data/glacier_pig/perf/{grid}/initial.nc")
        rows.append(
            dict(case="Solver scaling (Pine Island)", spacing_m=dx, nx=nx, ny=ny, nodes=nx * ny, basis="MOLHO, SSA",
                 probe_batch=9, newton_cg_role="cold solves (this figure); SSA uses 0")
        )
    rows.append(
        dict(case="Pablo Glacier (where 9 was chosen, not in the paper)", spacing_m=200.0, nx=227, ny=308,
             nodes=227 * 308, basis="MOLHO", probe_batch=9, newton_cg_role="probe_batch sweep 0/9/18/36")
    )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(f"{HERE}/data", exist_ok=True)
    for name, frame in (("solver_scaling", scaling()), ("probe_batch", probe_batch()), ("paper_case_grids", paper_cases())):
        frame.to_csv(f"{HERE}/data/{name}.csv", index=False, float_format="%.6g")
        print(f"== data/{name}.csv")
        print(frame.to_string(index=False))
