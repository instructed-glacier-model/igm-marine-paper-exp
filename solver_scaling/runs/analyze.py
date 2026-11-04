#!/usr/bin/env python3
"""Tables of the Newton-CG solver-scalability study (cold solves on a fixed PIG geometry).

Cases (cases/<name>/, written by pig_diag_bench.py through run_queue.sh):
  <res>_<basis>_<f32|f64>     main matrix: compile call + 5 cold solves
  <res>_molho_f32_pb9         the paper's probe_batch 9, to tie the curve to the paper's runs
  <res>_molho_f32_jacobi      optional control: MOLHO with point-block Jacobi
  split_<case>                one labelled cold solve with print_timing (Hessian / CG split)
  ref_<res>_<basis>, refB_<res>_<basis>   float64 tight references and their tighter continuation

Writes scaling.csv (one row per timed case), reference_stability.csv, cases/<case>/summary.json.
Accuracy: relative RMSE of the depth-averaged velocity (vector) of every cold solve against the
case's own-basis float64 reference, over all ice / grounded / floating, the quasi-rigid band excluded.
"""

import csv
import json
import re
import statistics
from pathlib import Path

import netCDF4
import numpy as np

S = Path(__file__).resolve().parent
RES = ("4km", "2km", "1km", "500m")
WEIGHTS = {"molho": (0.2, 0.8), "ssa": (1.0,)}
NZ = {"molho": 2, "ssa": 1}
FLAG = 0.01


def bar(U, basis):
    return np.einsum("z,zyx->yx", np.asarray(WEIGHTS[basis]), U.astype(np.float64))


def masks(res):
    with netCDF4.Dataset(S / "data" / res / "initial.nc") as d:
        act = np.asarray(d["active_node"][:]) > 0
        gr = np.asarray(d["grounded_node"][:]) > 0
        band = np.asarray(d["bandmask"][:]) > 0
        nx, ny = d.dimensions["x"].size, d.dimensions["y"].size
    keep = act & ~band
    return dict(all=keep, grounded=keep & gr, floating=keep & ~gr), nx, ny, int(act.sum())


def rel_rmse(U, V, Ur, Vr, basis, m):
    u, v, ur, vr = bar(U, basis), bar(V, basis), bar(Ur, basis), bar(Vr, basis)
    e2 = (u - ur) ** 2 + (v - vr) ** 2
    r2 = ur**2 + vr**2
    return {k: float(np.sqrt(e2[s].mean() / r2[s].mean())) if s.any() else None for k, s in m.items()}


def reference(res, basis, stage="ref"):
    d = S / "cases" / f"{stage}_{res}_{basis}"
    if not (d / "reference.npz").exists() or not (d / "summary_raw.json").exists():
        return None
    if not all(r["converged"] for r in json.load(open(d / "summary_raw.json"))["rows"]):
        return None
    return dict(np.load(d / "reference.npz"))


def timing_split(name):
    """Median per-Newton-iteration total and CG seconds from a split_<name> log (print_timing)."""
    log = S / "work" / "logs" / f"split_{name}.log"
    if not log.exists():
        return None
    rows = re.findall(r"\[timing\] iter=\s*(\d+) cg=([\d.]+)s\((\d+)it.*?total=([\d.]+)s", log.read_text())
    # the split case runs the compile call first (its iterations are traced): keep the last solve only
    solves, cur = [], []
    for it, cg, ncg, tot in rows:
        if int(it) == 0 and cur:
            solves.append(cur)
            cur = []
        cur.append((float(cg), int(ncg), float(tot)))
    solves.append(cur)
    last = solves[-1]
    if not last:
        return None
    return dict(
        newton_its=len(last),
        total_per_newton_s=statistics.median(t for _, _, t in last),
        cg_per_newton_s=statistics.median(c for c, _, _ in last),
        hessian_and_linesearch_per_newton_s=statistics.median(t - c for c, _, t in last),
    )


def main():
    rows_out, stab = [], []
    for res in RES:
        m, nx, ny, nact = masks(res)
        for basis in ("molho", "ssa"):
            ra, rb = reference(res, basis), reference(res, basis, "refB")
            if ra is not None and rb is not None:
                e = rel_rmse(ra["U_initial"], ra["V_initial"], rb["U_initial"], rb["V_initial"], basis, m)
                stab.append(dict(res=res, basis=basis, rel_rmse_A_vs_B=e["all"]))
            names = [f"{res}_{basis}_f32", f"{res}_{basis}_f64"]
            if basis == "molho":
                names += [f"{res}_molho_f32_pb9", f"{res}_molho_f32_jacobi"]
            for name in names:
                cdir = S / "cases" / name
                if not (cdir / "summary_raw.json").exists():
                    continue
                s = json.load(open(cdir / "summary_raw.json"))
                comp = [r for r in s["rows"] if r["phase"] == "compile"][0]
                cold = [r for r in s["rows"] if r["phase"] == "cold"]
                ok = [r for r in cold if r["converged"]]
                acc = []
                for r in ok:
                    npz = S / "work" / "raw" / name / f"cold_{r['rep']}.npz"
                    if ra is not None and npz.exists():
                        a = np.load(npz)
                        acc.append(rel_rmse(a["U"], a["V"], ra["U_initial"], ra["V_initial"], basis, m))
                med = lambda k: statistics.median(x[k] for x in acc) if acc else None  # noqa: E731
                t = [r["wall_s"] for r in ok]
                per_newton = [r["wall_s"] / r["newton_its"] for r in ok if r["newton_its"]]
                cg_per_newton = [r["cg_its_total"] / r["newton_its"] for r in ok if r["newton_its"]]
                split = timing_split(name)
                row = dict(
                    case=name,
                    res=res,
                    spacing_m=int(float(s["dofs"]["initial"]["spacing_m"])),
                    basis=basis,
                    precision="float64" if name.split("_")[2] == "f64" else "float32",
                    variant="paper probe_batch 9" if name.endswith("pb9") else ("jacobi" if name.endswith("jacobi") else "main"),
                    preconditioner=s["setup"]["preconditioner"],
                    probe_batch=s["settings"]["probe_batch"],
                    nx=nx,
                    ny=ny,
                    dofs_raster=2 * NZ[basis] * nx * ny,
                    dofs_active_ice=2 * NZ[basis] * nact,
                    compile_call_s=comp["wall_s"],
                    compile_status=comp["status"],
                    cold_converged=f"{len(ok)}/{len(cold)}",
                    cold_median_s=statistics.median(t) if t else None,
                    cold_min_s=min(t) if t else None,
                    cold_max_s=max(t) if t else None,
                    newton_its=statistics.median(r["newton_its"] for r in ok) if ok else None,
                    cg_per_newton=statistics.median(cg_per_newton) if cg_per_newton else None,
                    time_per_newton_s=statistics.median(per_newton) if per_newton else None,
                    tf_peak_mib=max([comp["tf_peak_mib"] or 0] + [r["tf_peak_mib"] or 0 for r in cold]),
                    smi_peak_mib=s["nvidia_smi_peak_mib"],
                    relrmse_all=med("all"),
                    relrmse_grounded=med("grounded"),
                    relrmse_floating=med("floating"),
                    flagged=sum(1 for x in acc if max(v for v in x.values() if v is not None) > FLAG),
                    split_hessian_per_newton_s=split["hessian_and_linesearch_per_newton_s"] if split else None,
                    split_cg_per_newton_s=split["cg_per_newton_s"] if split else None,
                    error=(comp["error"] or "")[:80],
                )
                rows_out.append(row)
                with open(cdir / "summary.json", "w") as f:
                    json.dump(dict(row=row, accuracy=acc, split=split, environment=s["environment"],
                                   settings=s["settings"], setup=s["setup"]), f, indent=1, default=str)
    for path, rows in ((S / "results" / "scaling.csv", rows_out), (S / "results" / "reference_stability.csv", stab)):
        if rows:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    for r in rows_out:
        print(r["case"], r["cold_converged"], r["cold_median_s"], r["newton_its"], r["cg_per_newton"],
              r["time_per_newton_s"], r["tf_peak_mib"], r["relrmse_all"])


if __name__ == "__main__":
    main()
