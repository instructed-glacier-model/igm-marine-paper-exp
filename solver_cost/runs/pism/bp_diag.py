#!/usr/bin/env python3
"""Fixed-geometry PISM SSA/BP benchmark. Run with the isolated environment's MPI.

All timings cover StressBalance.update(inputs, True), with a fresh solver per
repetition. Construction/init, input conversion, gathering and writing are separate.
The static enthalpy field encodes A; no thermodynamics or geometry is advanced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time

import numpy as np
import xarray as xr
from mpi4py import MPI

DRIVER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--model', choices=['ssa', 'bp'], default='ssa')
    p.add_argument('--nz', type=int, default=5)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--rtol', type=float, default=1e-6)
    p.add_argument('--ksp-rtol', type=float, default=1e-4)
    p.add_argument('--check', choices=['sliding', 'shelf'])
    p.add_argument('--front-treatment',choices=['native','igm-wet'],default='igm-wet')
    p.add_argument('--bp-preconditioner',choices=['sor','gamg'],default='gamg')
    p.add_argument('--pc', choices=['gamg', 'asm', 'lu'], default='gamg')
    p.add_argument('--no-cfbc', action='store_true')
    p.add_argument('--band-bc', choices=['native','stationary'], default='native')
    p.add_argument('--geometry-mode', choices=['native','ice-base'], default='native',
                   help='ice-base encodes the IGM diagnostic lower surface and nodal friction mask')
    args, extra = p.parse_known_args()
    # Only PETSc/PISM options should reach its command-line parser.
    sys.argv = [sys.argv[0]] + extra
    return args


def load_fields(args):
    if args.check:
        n = 41
        x = np.linspace(0., 100000., n)
        y = np.linspace(0., 100000., n)
        X, Y = np.meshgrid(x, y)
        a = np.full_like(X, 11.)
        h = np.full_like(X, 500.)
        if args.check == 'sliding':
            bed = np.full_like(X, 1000.)
            wl = np.full_like(X, -10000.)
            tau = np.full_like(X, .1)
            surf = bed + h
        else:
            # Straight shelf; prescribe inflow at x=0, free front at x=L.
            bed = np.full_like(X, -1000.)
            wl = np.zeros_like(X)
            tau = np.zeros_like(X)
            h[:, -2:] = 0.
            surf = h * (1. - 917./1027.)
        return dict(x=x, y=y, thk=h, topg=bed, water_level=wl,
                    usurf=surf, arrhenius=a, tau_ref=tau,
                    bandmask=np.zeros_like(X), domainmask=np.ones_like(X))
    with xr.open_dataset(args.initial) as d:
        return {k: d[k].values.copy() for k in
                ('x', 'y', 'thk', 'topg', 'water_level', 'usurf',
                 'arrhenius', 'tau_ref', 'bandmask', 'domainmask', 'active_node')}


def main():
    args = parse_args()
    import PISM
    sys.path.insert(0,str(Path(__file__).resolve().parent/'bp_front_extension'))
    from bp_front_boundary import FrontMatchedBlatter
    ctx = PISM.Context()
    cfg = ctx.config
    comm = MPI.COMM_WORLD
    rank = comm.rank
    args.out.mkdir(parents=True, exist_ok=True) if rank == 0 else None
    comm.Barrier()
    t_setup = time.perf_counter()
    f = load_fields(args)
    f['solver_bed'] = f['topg'].astype(float).copy()
    f['solver_tau_ref'] = f['tau_ref'].astype(float).copy()
    if args.geometry_mode == 'ice-base' and not args.check:
        # The seabed beneath a floating column does not enter a diagnostic solve.
        # Encode IGM's actual lower ice surface, so the interpolation of its
        # upper surface and of its nodal grounded-drag field is retained near GL.
        bottom=f['usurf'].astype(float)-f['thk']
        phi=f['thk']+(1027./917.)*(bottom-f['water_level'])
        tol=(64.*2.**-23)*(f['thk']+abs(bottom))
        grounded=phi>tol
        floating=(f['thk']>0)&~grounded
        f['solver_bed'][floating]=(f['water_level'].astype(float)[floating]
                                 -(917./1027.)*f['thk'].astype(float)[floating]-1e-7)
        f['solver_tau_ref'][~grounded]=0.
    # One year is the unit used by PISM/UDUNITS, recorded explicitly.
    year = PISM.util.convert(1., 'year', 'second')
    numbers = {
        'constants.ice.density': 917., 'constants.sea_water.density': 1027.,
        'constants.standard_gravity': 9.81,
        'basal_resistance.pseudo_plastic.q': 1./3.,
        'basal_resistance.pseudo_plastic.u_threshold': 100.,
        'basal_resistance.plastic.regularization': 1.,
        'basal_resistance.pseudo_plastic.sliding_scale_factor': 0.,
        'stress_balance.ssa.epsilon': 0.,
        'stress_balance.ssa.Glen_exponent': 3.,
        'stress_balance.blatter.Glen_exponent': 3.,
        'stress_balance.ssa.enhancement_factor': 1.,
        'stress_balance.blatter.enhancement_factor': 1.,
        'stress_balance.blatter.relative_convergence': args.ksp_rtol,
        'flow_law.Schoof_regularizing_length': .001,  # config unit is km
        'flow_law.Schoof_regularizing_velocity': 1e-5,
    }
    flags = {'basal_resistance.pseudo_plastic.enabled': True,
             'stress_balance.calving_front_stress_bc': not args.no_cfbc,
             'stress_balance.ssa.dirichlet_bc': bool(args.check == 'shelf' or args.band_bc == 'stationary'),
             'stress_balance.ssa.compute_surface_gradient_inward': False}
    strings = {'stress_balance.ssa.flow_law': 'arr',
               'stress_balance.blatter.flow_law': 'arr',
               'stress_balance.ssa.method': 'fem'}
    for k, v in numbers.items():
        cfg.set_number(k, v)
    for k, v in flags.items():
        cfg.set_flag(k, v)
    for k, v in strings.items():
        cfg.set_string(k, v)
    ice=f['thk']>0
    lower=f['usurf'].astype(float)-f['thk']
    land=(~ice)&(lower>f['water_level'])
    cell_ice=ice[:-1,:-1]&ice[1:,:-1]&ice[:-1,1:]&ice[1:,1:]
    cell_land=land[:-1,:-1]|land[1:,:-1]|land[:-1,1:]|land[1:,1:]
    wet=(~cell_ice)&(~cell_land)
    padded=np.pad(wet,1,constant_values=False)
    # PISM face ordering: left, right, front (-y), back (+y).
    fronts=np.stack([padded[1:-1,:-2],padded[1:-1,2:],
                     padded[:-2,1:-1],padded[2:,1:-1]],axis=-1)
    fronts &= cell_ice[...,None]
    x, y = f['x'], f['y']
    P = PISM.GridParameters(cfg)
    P.Mx, P.My = len(x), len(y)
    P.Lx, P.Ly = float(np.ptp(x)/2), float(np.ptp(y)/2)
    P.x0, P.y0 = float((x[0]+x[-1])/2), float((y[0]+y[-1])/2)
    P.registration = PISM.CELL_CORNER
    P.periodicity = (PISM.XY_PERIODIC if args.check == 'sliding' else
                     PISM.Y_PERIODIC if args.check else PISM.NOT_PERIODIC)
    P.z = PISM.DoubleVector([0., max(1., float(f['thk'].max())) + 1.])
    try:
        P.ownership_ranges_from_options(cfg, ctx.size)
    except TypeError:
        P.ownership_ranges_from_options(ctx.size)
    grid = PISM.Grid(ctx.ctx, P)
    geom = PISM.Geometry(grid)
    tauc = PISM.Scalar(grid, 'tauc')
    E = PISM.Array3D(grid, 'enthalpy', PISM.WITHOUT_GHOSTS, grid.z())
    # Solvers use the converter cached inside Context, so use that exact
    # instance (not a new converter after changing configuration values).
    EC = ctx.enthalpy_converter
    law = PISM.PatersonBuddCold(3., cfg, EC)
    # Invert the exact PISM cold Glen law for each distinct input A.
    temperatures = {}
    max_a_error = 0.
    beta = cfg.get_number('constants.ice.beta_Clausius_Clapeyron')
    p_air = EC.pressure(0.)
    pressure_gradient = EC.pressure(1.) - p_air
    zmax = float(grid.z()[-1])

    def column(T_pa, H):
        # Enthalpy is linear in z for constant heat capacity. Extrapolate the
        # above-ice endpoint so interpolation inside the ice is exact, rather
        # than clamping it at the surface. These are artificial static fields.
        return [EC.enthalpy(T_pa-beta*(p_air+pressure_gradient*(H-z)),0.,p_air)
                for z in (0.,zmax)]

    for A in np.unique(f['arrhenius']):
        a_si = float(A) * 1e-18 / year
        temp = law.tempFromSoftness(a_si)
        temperatures[float(A)] = temp
        for H in (0.,float(f['thk'].max())):
            e0,e1 = column(temp,H)
            for depth in np.linspace(0.,H,5):
                z=H-depth
                e=e0+(e1-e0)*z/zmax
                max_a_error=max(max_a_error,abs(law.softness(e,EC.pressure(depth))/a_si-1.))
    if max_a_error > 1e-10:
        raise RuntimeError(f'Rheology mapping failed: {max_a_error}')
    fields = [(geom.ice_thickness, 'thk'), (geom.bed_elevation, 'solver_bed'),
              (geom.sea_level_elevation, 'water_level')]
    with PISM.vec.Access([v for v, _ in fields] + [tauc, E]):
        for i, j in grid.points():
            for v, key in fields:
                v[i, j] = float(f[key][j, i])
            tauc[i, j] = float(f['solver_tau_ref'][j, i]) * 1e6
            E.set_column(i, j, column(temperatures[float(f['arrhenius'][j, i])],float(f['thk'][j,i])))
    for v, _ in fields:
        v.update_ghosts()
    geom.ensure_consistency(0.)
    if args.check == 'sliding':
        # Constant prescribed driving stress isolates the basal law on a flat,
        # periodic bed with no membrane deformation.
        driving_x = PISM.Scalar(grid, 'ssa_driving_stress_x')
        driving_y = PISM.Scalar(grid, 'ssa_driving_stress_y')
        driving_x.set(917. * 9.81 * 500. * .01)
        driving_y.set(0.)
        grid.variables().add(driving_x)
        grid.variables().add(driving_y)
    inputs = PISM.StressBalanceInputs()
    inputs.geometry = geom
    inputs.basal_yield_stress = tauc
    inputs.enthalpy = E
    melange = PISM.Scalar(grid, 'melange_back_pressure')
    melange.set(0.)
    inputs.melange_back_pressure = melange
    # Manufactured checks only: explicit inflow avoids a free rigid translation.
    bc_mask, bc_vel = PISM.Scalar1(grid, 'vel_bc_mask'), PISM.Vector1(grid, 'vel_bc')
    bc_mask.set(0.)
    bc_vel.set(0.)
    if args.check == 'shelf' or args.band_bc == 'stationary':
        with PISM.vec.Access([bc_mask, bc_vel]):
            for i, j in grid.points():
                if args.check == 'shelf' and i == 0:
                    bc_mask[i, j] = 1.
                    bc_vel.setitem(i, j, 0. if args.model=='bp' else 100./year, 0.)
                elif args.band_bc == 'stationary' and f['bandmask'][j, i] > .5:
                    bc_mask[i, j] = 1.
        bc_mask.update_ghosts()
        bc_vel.update_ghosts()
        inputs.bc_mask, inputs.bc_values = bc_mask, bc_vel
    opts = PISM.PETSc.Options()
    prefix = '' if args.model == 'ssa' else 'bp_'
    petsc = {
        f'{prefix}snes_rtol': args.rtol, f'{prefix}snes_atol': 1e-12,
        f'{prefix}snes_stol': 1e-12, f'{prefix}snes_max_it': 200,
        f'{prefix}ksp_rtol': args.ksp_rtol, f'{prefix}ksp_max_it': 2000,
        f'{prefix}ksp_type': 'gmres',
        f'{prefix}snes_converged_reason': None,
        f'{prefix}ksp_converged_reason': None,
    }
    if args.model == 'ssa':
        petsc['pc_type'] = args.pc
        if args.pc == 'gamg':
            petsc['pc_gamg_type'] = 'agg'
        if args.pc == 'lu':
            petsc['pc_factor_mat_solver_type'] = 'mumps'
    elif args.nz == 2 and args.bp_preconditioner == 'gamg':
        # There is no vertical coarsening with a single vertical element.
        # PETSc's one-level PCMG uses mg_levels_0_ instead of mg_coarse_.
        # Use algebraic multigrid directly on this system.
        petsc.update({'bp_pc_type': 'gamg', 'bp_pc_gamg_type': 'agg'})
    else:
        levels = int(np.log2(args.nz-1)) + 1
        petsc.update({'bp_pc_type': 'mg', 'bp_pc_mg_levels': levels,
                      'bp_mg_levels_ksp_type': 'richardson',
                      'bp_mg_levels_ksp_max_it': 2,
                      'bp_mg_levels_pc_type': 'sor',
                      'bp_mg_coarse_ksp_type': 'preonly',
                      'bp_mg_coarse_pc_type': 'gamg'})
    for k, v in petsc.items():
        opts.setValue(k, v)
    cell_type = geom.cell_type.to_numpy()
    surface = geom.ice_surface_elevation.to_numpy()
    if rank == 0:
        ice = f['thk'] > 0
        bottom = f['usurf'].astype(float) - f['thk']
        phi = f['thk'] + (1027./917.) * (bottom-f['water_level'])
        gr = ice & (phi > (64.*2.**-23)*(f['thk']+abs(bottom)))
        audit = dict(front_treatment=args.front_treatment,
                     prescribed_wet_front_faces=int(fronts.sum()),
                     rate_factor_max_relative_error=max_a_error,
                     geometry_mode=args.geometry_mode,
                     raised_floating_bed_nodes=int((abs(f['solver_bed']-f['topg'])>1e-6).sum()),
                     surface_max_abs_difference_m=float(np.max(abs(surface[ice]-f['usurf'][ice]))),
                     ice_nodes=int(ice.sum()), band_nodes=int((ice & (f['bandmask'] > 0)).sum()),
                     igm_grounded_nodes=int(gr.sum()),
                     pism_grounded_nodes=int((cell_type == 2).sum()),
                     grounded_mask_disagreements=int(((cell_type == 2) != gr)[ice].sum()),
                     x_max_abs_difference_m=float(np.max(abs(np.array(grid.x())-x))),
                     y_max_abs_difference_m=float(np.max(abs(np.array(grid.y())-y))))
        (args.out/'audit.json').write_text(json.dumps(audit, indent=2))
        xr.Dataset({'pism_cell_type': (('y','x'), cell_type),
                    'pism_surface': (('y','x'), surface),
                    'physical_bed': (('y','x'),f['topg']),
                    'solver_bed': (('y','x'),f['solver_bed']),
                    'solver_tauc_pa': (('y','x'),f['solver_tau_ref']*1e6),
                    'marine_front_allowed': (('y_cell','x_cell','face'),fronts.astype(np.int8))},
                   coords={'x':x,'y':y,'x_cell':.5*(x[:-1]+x[1:]),
                           'y_cell':.5*(y[:-1]+y[1:]),'face':['W','E','S','N']}).to_netcdf(args.out/'geometry.nc')
        print('AUDIT', json.dumps(audit), flush=True)
    setup_s = comm.allreduce(time.perf_counter()-t_setup, op=MPI.MAX)
    rows = []
    for rep in range(args.reps + 1):
        phase = 'initialization' if rep == 0 else 'cold'
        if rank == 0:
            print(f'BEGIN_SOLVE {phase} {rep}', flush=True)
        t_construct = time.perf_counter()
        if args.model == 'ssa':
            model = PISM.SSAFEM(grid)
        else:
            fixed=np.zeros_like(f['thk'],dtype=np.int32)
            if args.band_bc=='stationary':
                fixed[f['bandmask']>.5]=1
            if args.check=='shelf':
                fixed[:,0]=1
                # The adapter currently prescribes zero; the shelf check must
                # use zero inflow in its analytic reference.
            model=FrontMatchedBlatter(grid,args.nz,2,fixed.ravel().tolist(),
                fronts.astype(np.int32).ravel().tolist() if args.front_treatment=='igm-wet' else [])

        if args.model == 'ssa' and args.pc == 'gamg':
            # SSAFEM selects BAIJ in its constructor; GAMG requires AIJ.
            # Vector defaults to stencil width 2 even without ghost storage.
            grid.get_dm(2, 2).get().setMatType('aij')
        model.init()
        construct_s = comm.allreduce(time.perf_counter()-t_construct, op=MPI.MAX)
        comm.Barrier()
        t0 = MPI.Wtime()
        error = None
        try:
            model.update(inputs, True)
        except Exception as e:
            error = repr(e)
        comm.Barrier()
        wall = comm.allreduce(MPI.Wtime()-t0, op=MPI.MAX)
        errors = comm.allgather(error)
        ok = not any(errors)
        rss = comm.allgather(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        row = dict(phase=phase, rep=rep, wall_s=wall, construct_s=construct_s,
                   converged=ok, errors=errors if not ok else [],
                   peak_rank_rss_bytes=max(rss), sum_rank_peak_rss_bytes=sum(rss))
        if ok:
            if args.model == 'ssa':
                vel = model.velocity().to_numpy()
                if rank == 0:
                    u, v = vel[..., 0]*year, vel[..., 1]*year
                    vals = dict(ubar=u, vbar=v, uvelsurf=u, vvelsurf=v, uvelbase=u, vvelbase=v)
            else:
                u3 = model.velocity_u_sigma().to_numpy()
                v3 = model.velocity_v_sigma().to_numpy()
                if rank == 0:
                    u3, v3 = u3*year, v3*year
                    vals = dict(uvelsurf=u3[..., -1], vvelsurf=v3[..., -1],
                                uvelbase=u3[..., 0], vvelbase=v3[..., 0],
                                ubar=np.trapezoid(u3, dx=1./(args.nz-1), axis=-1),
                                vbar=np.trapezoid(v3, dx=1./(args.nz-1), axis=-1))
            if rank == 0:
                row['finite'] = all(bool(np.isfinite(a).all()) for a in vals.values())
                row['max_speed_m_yr'] = float(np.hypot(vals['ubar'], vals['vbar']).max())
                ds = xr.Dataset({k: (('y','x'), a) for k, a in vals.items()}, coords={'x': x, 'y': y})
                for k in ds.data_vars:
                    ds[k].attrs['units'] = 'm yr-1'
                ds.to_netcdf(args.out/f'{phase}_{rep}.nc')
        rows.append(row)
        if rank == 0:
            print('END_SOLVE', json.dumps(row), flush=True)
            summary = dict(model=args.model, nz=1 if args.model == 'ssa' else args.nz,
                ranks=comm.size, grid_shape=[len(y), len(x)], spacing_m=float(x[1]-x[0]),
                allocated_velocity_dofs=2*len(x)*len(y)*(1 if args.model == 'ssa' else args.nz),
                initial=str(args.initial), input_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest() if args.initial else None,
                driver_sha256=DRIVER_SHA256,
                bp_extension_sha256=hashlib.sha256((Path(__file__).resolve().parent/'bp_front_extension/_bp_front_boundary.so').read_bytes()).hexdigest(),
                bp_boundary_implementation='C++ subclass: stationary band and IGM wet-cell marine-front classification',
                front_treatment=args.front_treatment,bp_preconditioner=args.bp_preconditioner,
                check=args.check, band_bc=args.band_bc, geometry_mode=args.geometry_mode,
                year_seconds=year, setup_s=setup_s, rows=rows,
                settings=dict(numbers=numbers, flags=flags, strings=strings, petsc=petsc),
                environment=dict(host=socket.gethostname(), python=platform.python_version(),
                                 pism_version=str(getattr(PISM, '__version__', 'see environment lock')),
                                 mpi=MPI.Get_library_version(), executable=sys.executable,
                                 argv=sys.argv, thread_env={k:os.getenv(k) for k in
                                 ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
            (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
        del model
        if not ok:
            break
    if not rows[-1]['converged']:
        raise RuntimeError('Stress balance failed; see saved summary')


if __name__ == '__main__':
    main()
