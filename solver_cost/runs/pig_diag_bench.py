#!/usr/bin/env python3
"""Diagnostic-only IGM Newton-CG benchmark on the shared PIG grids.

One process runs one case. Only IGM's iceflow module is initialised; thickness, time, fronts,
melt and SMB are never advanced. The solver is IGM's own (identity mapping + cg_newton), called
through igm.processes.iceflow.unified.solver.solve_iceflow exactly as a time step calls it.

Modes
  timing     setup (iceflow initialisation without its built-in solve), one compile call from
             zero velocity, N compiled cold solves (velocity reset to zero before each) and N warm
             solves on the changed geometry (each from this case's own initial-state solution).
             The raw U, V of every repetition are saved for the accuracy step (analyze.py).
  reference  tight solves on the initial and the changed geometry (igm-marine snapshot.yaml
             settings by default: float64, CG tol 1e-4, 0.1 m/yr halt, 400 Newton iterations).
             --start-from continues from an earlier reference (stability check, tighter settings).
  native     the native unpadded 1-km check: S0 (cold, nbit_init) then S1 (warm re-solve on the
             same geometry), as in the saved run; compared with frame 0 by analyze.py.

Run with the igm checkout first on PYTHONPATH (igm 058558bc + the run's local patch).
"""

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

FIELDS = ("thk", "usurf", "topg", "water_level", "arrhenius", "tau_ref")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=("timing", "reference", "native"), required=True)
    p.add_argument("--basis", choices=("molho", "ssa"), required=True)
    p.add_argument("--precision", choices=("single", "double"), default="single")
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--changed", type=Path)
    p.add_argument("--config", type=Path, required=True, help="resolved run config")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--raw", type=Path, help="folder for the raw U, V of each solve")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--start-from", type=Path, help="reference mode: npz of a reference")
    p.add_argument("--cg-tol", type=float)
    p.add_argument("--halt-tol", type=float)
    p.add_argument("--nbit-init", type=int)
    p.add_argument("--probe-batch", type=int)
    p.add_argument("--gpu-index", type=int, help="physical GPU index (nvidia-smi)")
    return p.parse_args()


class GpuSampler:
    """memory.used of one GPU from nvidia-smi every 100 ms (whole device, MiB)."""

    def __init__(self, index):
        self.samples = []
        self.proc = None
        if index is None:
            return
        cmd = [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "--loop-ms=100",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.samples.append((time.perf_counter(), float(line.strip())))
            except ValueError:
                pass

    def peak(self, t0=None, t1=None):
        vals = [
            m
            for t, m in self.samples
            if (t0 is None or t >= t0) and (t1 is None or t <= t1)
        ]
        if vals:
            return max(vals)
        return None

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cfg(args):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    # Fill in the options added to IGM since the paper configuration was resolved, from the installed IGM;
    # every explicit setting of the configuration keeps precedence.
    import igm

    defaults = OmegaConf.load(Path(igm.__file__).parent / "conf/processes/iceflow.yaml").iceflow
    cfg.processes.iceflow = OmegaConf.merge(defaults, cfg.processes.iceflow)
    ice = cfg.processes.iceflow
    uni = ice.unified
    cg = uni.cg_newton
    ice.numerics.precision = args.precision
    if args.basis == "ssa":
        ice.numerics.basis_vertical = "ssa"
        ice.numerics.Nz = 1
        cg.preconditioner = "block_jacobi"
        cg.probe_batch = 0  # the SSA banded operator rejects probe_batch > 0
    else:
        if ice.numerics.basis_vertical != "molho" or ice.numerics.Nz != 2:
            raise SystemExit("the run config is not MOLHO Nz = 2")
        if cg.preconditioner != "barotropic_multigrid":
            raise SystemExit("the run config does not use barotropic_multigrid")
    cg.print_timing = False
    uni.display.print_cost = False
    if args.mode == "reference":
        ice.numerics.precision = "double"
        uni.nbit_init = 400
        cg.cg_tol = 1.0e-4
        cg.cold_restart_interval = 0.0
        if args.basis == "molho":
            cg.probe_batch = 3
        uni.halt.success[0].abs_change.tol = 0.1
    if args.cg_tol is not None:
        cg.cg_tol = args.cg_tol
    if args.halt_tol is not None:
        uni.halt.success[0].abs_change.tol = args.halt_tol
    if args.nbit_init is not None:
        uni.nbit_init = args.nbit_init
    if args.probe_batch is not None and args.basis == "molho":
        cg.probe_batch = args.probe_batch
    if uni.halt.success[0].criterion != "abs_change":
        raise SystemExit("unexpected halt criterion")
    cfg.core.folder_data = ""
    return cfg


def load_state(cfg, path):
    from igm.common import State
    from igm.inputs import local as local_input

    state = State()
    state.original_cwd = path.parent.resolve()
    state.saveresult = True
    cfg.inputs.local.filename = str(path.resolve())
    local_input.run(cfg, state)
    return state


class Recorder:
    """Per-solve counters, collected without extra device synchronisation."""

    def __init__(self, opt):
        self.cg = []
        self.alpha = []
        self.halt = []
        cg_solve = opt._cg_solve
        line_search = opt._line_search
        finalize = opt._finalize_display

        def cg_wrapped(*a, **k):
            out = cg_solve(*a, **k)
            self.cg.append((out[1], out[2]))
            return out

        def ls_wrapped(*a, **k):
            alpha = line_search(*a, **k)
            self.alpha.append(alpha)
            return alpha

        def fin_wrapped(halt_status):
            self.halt.append(halt_status)
            return finalize(halt_status)

        opt._cg_solve = cg_wrapped
        opt._line_search = ls_wrapped
        opt._finalize_display = fin_wrapped

    def start(self):
        self.cg, self.alpha, self.halt = [], [], []

    def collect(self):
        from igm.processes.iceflow.unified.halt import HaltStatus

        its = [int(i.numpy()) for i, _ in self.cg]
        res = [float(r.numpy()) for _, r in self.cg]
        alphas = [float(a.numpy()) for a in self.alpha]
        stop = None
        if self.halt:
            stop = HaltStatus(int(self.halt[-1].numpy())).name
        return dict(
            newton_its=len(its),
            cg_its=its,
            cg_relres=res,
            alphas=alphas,
            stop=stop,
        )


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.raw is not None:
        args.raw.mkdir(parents=True, exist_ok=True)
    sampler = GpuSampler(args.gpu_index)
    time.sleep(0.5)
    smi_baseline = sampler.peak()

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    import igm
    from igm.common import State
    from igm.common.aliases import load_aliases_from_core_cfg
    import igm.processes.iceflow.iceflow as iceflow_mod
    import igm.processes.iceflow.unified.unified as unified_mod
    from igm.processes.thk.masks import (
        compute_grounded_mask,
        compute_cell_ice_mask,
        iceflow_node_mask,
    )

    cfg = build_cfg(args)
    State.register_aliases(load_aliases_from_core_cfg(cfg.core.aliases))
    rho_ratio = (
        cfg.processes.iceflow.physics.water_density
        / cfg.processes.iceflow.physics.ice_density
    )
    devices = tf.config.list_physical_devices("GPU")
    details = tf.config.experimental.get_device_details(devices[0]) if devices else {}
    build = tf.sysconfig.get_build_info()
    env = dict(
        host=socket.gethostname(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_index_physical=args.gpu_index,
        gpu=details.get("device_name"),
        compute_capability=details.get("compute_capability"),
        python=platform.python_version(),
        tensorflow=tf.__version__,
        tf_cuda_build=build.get("cuda_version"),
        tf_cudnn_build=build.get("cudnn_version"),
        igm_imported=os.path.dirname(igm.__file__),
        driver_sha256=sha256(__file__),
        argv=sys.argv,
        nvidia_smi_baseline_mib=smi_baseline,
    )

    def tf_peak_mib():
        if not devices:
            return None
        return tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20

    def tf_reset():
        if devices:
            tf.config.experimental.reset_memory_stats("GPU:0")

    # setup: inputs + iceflow initialisation without the built-in first solve
    t_setup0 = time.perf_counter()
    state = load_state(cfg, args.initial)
    t_load = time.perf_counter()
    other = None
    if args.changed is not None:
        other = load_state(copy.deepcopy(cfg), args.changed)
    t_load2 = time.perf_counter()
    real_solve = unified_mod.solve_iceflow
    unified_mod.solve_iceflow = lambda cfg, state, init=False: None
    tf_reset()
    t_init0 = time.perf_counter()
    iceflow_mod.initialize(cfg, state)
    opt = state.iceflow.optimizer
    mapping = state.iceflow.mapping
    discr_v = state.iceflow.discr_v

    def sync():
        tf.reshape(mapping.U, [-1])[0].numpy()
        tf.reshape(mapping.V, [-1])[0].numpy()
        tf.reshape(state.thk, [-1])[0].numpy()

    sync()
    t_init1 = time.perf_counter()
    unified_mod.solve_iceflow = real_solve
    setup = dict(
        load_initial_s=t_load - t_setup0,
        load_changed_s=t_load2 - t_load,
        iceflow_initialize_s=t_init1 - t_init0,
        tf_peak_mib=tf_peak_mib(),
        preconditioner=type(opt.preconditioner).__name__,
        operator=type(opt.operator).__name__,
        precision=str(opt.precision),
    )
    rec = Recorder(opt)

    def geometry(src):
        for f in FIELDS:
            getattr(state, f).assign(getattr(src, f))

    initial_fields = {f: tf.identity(getattr(state, f)) for f in FIELDS}

    class _Src:
        pass

    initial = _Src()
    for f in FIELDS:
        setattr(initial, f, initial_fields[f])

    def masks():
        thk, usurf, wl = state.thk, state.usurf, state.water_level
        grounded = compute_grounded_mask(thk, usurf - thk, wl, rho_ratio)
        cells = compute_cell_ice_mask(thk, grounded).numpy()
        nodes = iceflow_node_mask(thk, usurf, wl, rho_ratio).numpy()
        support = np.zeros(nodes.shape, bool)
        support[:-1, :-1] |= cells
        support[:-1, 1:] |= cells
        support[1:, :-1] |= cells
        support[1:, 1:] |= cells
        return dict(
            grounded=grounded.numpy() & (thk.numpy() > 0),
            cells=cells,
            nodes=nodes,
            support=support,
            ice=thk.numpy() > 0,
        )

    def dof_counts(m):
        nz = int(cfg.processes.iceflow.numerics.Nz)
        ny, nx = m["ice"].shape
        return dict(
            nx=nx,
            ny=ny,
            spacing_m=float(state.dx.numpy()),
            Nz=nz,
            components_per_node=2 * nz,
            active_elements=int(m["cells"].sum()),
            supported_nodes=int(m["support"].sum()),
            ice_nodes=int(m["ice"].sum()),
            active_ice_nodes=int(m["nodes"].sum()),
            unsupported_ice_nodes=int((m["ice"] & ~m["nodes"]).sum()),
            supported_ice_free_nodes=int((m["support"] & ~m["ice"]).sum()),
            allocated_slots=2 * nz * nx * ny,
            active_ice_dofs=2 * nz * int(m["nodes"].sum()),
            energy_carrying_dofs=2 * nz * int(m["support"].sum()),
            fixed_slots=2 * nz * (nx * ny - int(m["support"].sum())),
        )

    def solve(init, label, rep):
        rec.start()
        tf_reset()
        sync()
        t0 = time.perf_counter()
        status, error = "ok", None
        try:
            real_solve(cfg, state, init=init)
        except Exception as e:  # failure criteria raise; keep the case going
            status, error = "exception", repr(e)
        sync()
        t1 = time.perf_counter()
        stats = rec.collect()
        U = mapping.U.numpy()[0].copy()
        V = mapping.V.numpy()[0].copy()
        finite = bool(np.isfinite(U).all() and np.isfinite(V).all())
        converged = status == "ok" and stats["stop"] == "SUCCESS" and finite
        row = dict(
            phase=label,
            rep=rep,
            wall_s=t1 - t0,
            status=status,
            stop=stats["stop"],
            converged=converged,
            finite=finite,
            newton_its=stats["newton_its"],
            cg_its_total=int(sum(stats["cg_its"])),
            cg_its=" ".join(str(i) for i in stats["cg_its"]),
            cg_relres_last=stats["cg_relres"][-1] if stats["cg_relres"] else None,
            zero_cg_steps=int(sum(1 for i in stats["cg_its"] if i == 0)),
            alpha_min=min(stats["alphas"]) if stats["alphas"] else None,
            alphas=" ".join(f"{a:.3g}" for a in stats["alphas"]),
            tf_peak_mib=tf_peak_mib(),
            smi_peak_mib=sampler.peak(t0, t1),
            max_abs_u=float(max(np.abs(U).max(), np.abs(V).max())),
            error=error,
        )
        if args.raw is not None:
            np.savez(args.raw / f"{label}_{rep}.npz", U=U, V=V)
        print(
            f"{label} {rep}: {row['wall_s']:.3f} s, {row['newton_its']} Newton, "
            f"{row['cg_its_total']} CG, stop {row['stop']}, status {status}, "
            f"alpha_min {row['alpha_min']}, max|u| {row['max_abs_u']:.1f}",
            flush=True,
        )
        return row, U, V

    def set_velocity(U, V):
        dtype = mapping.U.dtype
        mapping.set_theta(
            [
                tf.constant(U[np.newaxis], dtype=dtype),
                tf.constant(V[np.newaxis], dtype=dtype),
            ]
        )
        opt.reset_warm_start()

    shape = tuple(mapping.U.shape[1:])
    zeros = np.zeros(shape)
    m_init = masks()
    dofs = dict(initial=dof_counts(m_init))
    rows = []
    solutions = {}

    if args.mode in ("timing", "native"):
        state.it = 0
        set_velocity(zeros, zeros)
        row, U, V = solve(True, "compile", 0)
        rows.append(row)
        solutions["compile"] = (U, V)
    if args.mode == "native":
        state.it = 1
        row, U, V = solve(False, "native_S1", 1)
        rows.append(row)
        solutions["native_S1"] = (U, V)
    if args.mode == "timing":
        for rep in range(1, args.reps + 1):
            state.it = 0
            set_velocity(zeros, zeros)
            row, U, V = solve(True, "cold", rep)
            rows.append(row)
            solutions["cold"] = (U, V)
        if other is not None:
            U0, V0 = solutions["cold"]
            geometry(other)
            dofs["changed"] = dof_counts(masks())
            for rep in range(1, args.reps + 1):
                state.it = 1
                set_velocity(U0, V0)
                row, U, V = solve(False, "warm", rep)
                rows.append(row)
                solutions["warm"] = (U, V)
            geometry(initial)
    if args.mode == "reference":
        start = None
        if args.start_from is not None:
            start = np.load(args.start_from)
        state.it = 0
        if start is not None:
            set_velocity(start["U_initial"], start["V_initial"])
        else:
            set_velocity(zeros, zeros)
        row, U, V = solve(True, "reference_initial", 0)
        rows.append(row)
        solutions["reference_initial"] = (U, V)
        if other is not None:
            geometry(other)
            dofs["changed"] = dof_counts(masks())
            if start is not None:
                set_velocity(start["U_changed"], start["V_changed"])
            else:
                set_velocity(U, V)
            row, U, V = solve(True, "reference_changed", 0)
            rows.append(row)
            solutions["reference_changed"] = (U, V)
            geometry(initial)
        np.savez(
            args.out / "reference.npz",
            **{
                f"{n}_{k.split('_', 1)[1]}": a
                for k, (Uk, Vk) in solutions.items()
                for n, a in (("U", Uk), ("V", Vk))
            },
        )

    # velocity fields of the saved solutions (masked to the solver's ice nodes)
    Vb = discr_v.V_b.numpy().astype(np.float64)
    Vs = discr_v.V_s.numpy().astype(np.float64)
    Vbar = discr_v.V_bar.numpy().astype(np.float64)

    def write_solution(path, U, V, m, geometry_name):
        import netCDF4

        node = m["nodes"]
        with netCDF4.Dataset(path, "w") as d:
            d.title = "IGM Newton-CG diagnostic solution (pig_diag_bench.py)"
            d.geometry = geometry_name
            d.basis = args.basis
            d.precision = args.precision if args.mode != "reference" else "double"
            d.velocity_dtype = str(U.dtype)
            d.createDimension("y", U.shape[1])
            d.createDimension("x", U.shape[2])
            for name, vals in (("x", state.x.numpy()), ("y", state.y.numpy())):
                v = d.createVariable(name, "f8", (name,))
                v[:] = vals
                v.units = "m"
            for name, w, comp in (
                ("uvelsurf", Vs, U),
                ("vvelsurf", Vs, V),
                ("uvelbase", Vb, U),
                ("vvelbase", Vb, V),
                ("ubar", Vbar, U),
                ("vbar", Vbar, V),
            ):
                v = d.createVariable(name, U.dtype, ("y", "x"), zlib=True)
                v[:] = np.where(
                    node, np.einsum("z,zyx->yx", w, comp.astype(np.float64)), 0.0
                ).astype(U.dtype)
                v.units = "m/yr"
            for name in ("nodes", "grounded"):
                v = d.createVariable(f"mask_{name}", "i1", ("y", "x"), zlib=True)
                v[:] = m[name].astype(np.int8)

    if args.mode == "timing":
        write_solution(
            args.out / "initial_solution.nc", *solutions["cold"], m_init, "initial"
        )
        if "warm" in solutions:
            geometry(other)
            write_solution(
                args.out / "changed_solution.nc", *solutions["warm"], masks(), "changed"
            )
            geometry(initial)
    if args.mode == "native":
        for k in ("compile", "native_S1"):
            write_solution(
                args.out / f"{k}_solution.nc", *solutions[k], m_init, "initial"
            )

    from omegaconf import OmegaConf

    OmegaConf.save(cfg, args.out / "config.yaml")
    fields = list(rows[0].keys())
    with open(args.out / "repetitions_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    sampler.stop()
    summary = dict(
        mode=args.mode,
        basis=args.basis,
        precision=cfg.processes.iceflow.numerics.precision,
        initial=str(args.initial),
        initial_sha256=sha256(args.initial),
        changed=str(args.changed) if args.changed else None,
        changed_sha256=sha256(args.changed) if args.changed else None,
        settings=dict(
            nbit_init=int(cfg.processes.iceflow.unified.nbit_init),
            nbit=cfg.processes.iceflow.unified.nbit,
            cg_tol=float(cfg.processes.iceflow.unified.cg_newton.cg_tol),
            cg_max_iter=int(cfg.processes.iceflow.unified.cg_newton.cg_max_iter),
            probe_batch=int(cfg.processes.iceflow.unified.cg_newton.probe_batch),
            halt=OmegaConf.to_container(cfg.processes.iceflow.unified.halt.success),
            preconditioner_cfg=str(
                cfg.processes.iceflow.unified.cg_newton.preconditioner
            ),
        ),
        setup=setup,
        dofs=dofs,
        environment=env,
        nvidia_smi_peak_mib=sampler.peak(),
        rows=rows,
    )
    with open(args.out / "summary_raw.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("done", flush=True)


if __name__ == "__main__":
    main()
