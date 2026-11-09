# Comparison with PISM (paper Sect. 5.4, Appendix D, Fig. 7a–b, Table D1)

Velocity solves of PISM 2.3.1 on the four Pine Island grids of the scaling study (`../data/<res>/initial.nc`: 4, 2,
1 and 0.5 km), with no evolution of the thickness, melt, calving or front, against IGM's Newton–CG solver on the same
grids: PISM's SSA solver against IGM's SSA basis, and PISM's Blatter–Pattyn solver with two vertical levels, which
has the same number of velocity unknowns (4 N_x N_y) as IGM's MOLHO basis.

## Set-up

- **PISM** 2.3.1 (conda build `mpi_openmpi_hca9fe9e_0`), PETSc 3.24.6, Open MPI 5.0.10, Python 3.11; double
  precision; 32 MPI ranks, one per physical core, bound (`--bind-to core --map-by core`), one thread per rank, on a
  dual-socket Intel Xeon E5-2683 v4 node (2 × 16 cores, 2.1 GHz, 1 TiB). **IGM** ran on one NVIDIA RTX 4090.
- **Solvers.** SSA: Q1 finite elements. Blatter–Pattyn: two vertical levels (five and nine levels change the
  solution by 1.3–1.7 % and then by less than 0.15 %, so two levels are a cost-matched, not a converged,
  configuration). Both: Newton with GMRES and the GAMG algebraic multigrid of PETSc, relative tolerances 10⁻⁶
  (nonlinear) and 10⁻⁴ (linear); tightening them to 10⁻⁹ and 10⁻⁷ changes the velocities by less than 0.1 %. GAMG
  rebuilds its interpolation at every Newton iteration for BP, and for SSA at 0.5 km only. Every PETSc option of each
  case is in `results/resolved_pism_settings.json`.
- **Same physics as the IGM runs.** Geometry, water level, basal coefficients and the rate factor (through a static
  enthalpy field) of the input grids; ρ_i = 917, ρ_w = 1027 kg m⁻³, g = 9.81 m s⁻²; PISM's pseudo-plastic law with
  exponent 1/3, reference speed 100 m yr⁻¹ and regularization speed 1 m yr⁻¹, i.e. IGM's Weertman law without its
  bed-slope correction. Beneath floating ice the solver bed is the lower ice surface, and friction is masked with
  IGM's nodal grounding criterion (`--geometry-mode ice-base`; the agreement of the masks is in `audit.json`).
- **The quasi-rigid band and the front.** SSA: the band is a stationary boundary (`--band-bc stationary`). BP: the
  band keeps its finite drag, and a small subclass of PISM's solver (`bp_front_extension/`) takes IGM's
  classification of the marine front, so that the pressure acts on the same faces in both codes.
- **Timing.** One initialization solve, then five solves from zero velocity with a fresh solver state; the median is
  reported. The PISM timer covers the whole stress-balance update (assembly, Newton and Krylov iterations,
  preconditioner), synchronized across ranks; input, solver construction, process start-up and output are excluded,
  as the compilation is for IGM. Memory is the sum over the ranks of the peak resident set.
- **Velocity differences** against IGM (`results/velocity_differences_*.csv`, relative vector RMS over the ice
  without the band): SSA 0.9–4.9 %, BP 1.3–1.6 %.

## The eight cases (`results/pism_case_summary.csv`; Table D1)

| Model | Grid | Unknowns | PISM time (s) | Newton / Krylov iterations | PISM memory (GiB) |
|---|---|---:|---:|---:|---:|
| SSA | 4 km | 33 512 | 0.15 | 8 / 132 | 5.2 |
| | 2 km | 133 010 | 0.79 | 9 / 190 | 5.5 |
| | 1 km | 529 032 | 6.37 | 14 / 272 | 6.9 |
| | 0.5 km | 2 115 746 | 31.4 | 13 / 274 | 12.8 |
| BP, 2 levels | 4 km | 67 024 | 0.81 | 8 / 684 | 5.2 |
| | 2 km | 266 020 | 11.6 | 13 / 1623 | 5.8 |
| | 1 km | 1 058 064 | 28.4 | 13 / 835 | 8.1 |
| | 0.5 km | 4 231 492 | 100 | 13 / 686 | 17.2 |

The IGM times of Fig. 7a–b and Table D1 (`../../figure/data/cpu_gpu_comparison.csv`) were measured in the same
campaign as the PISM runs, with the IGM driver of the parent folder (`../pig_diag_bench.py`, IGM `b15cdce2`) and,
at 0.5 km for MOLHO, with `probe_batch 1` (nine probes per batch run out of GPU memory). They agree with the times
of the scaling study (`../../figure/data/solver_scaling.csv`, Fig. 7c) within 2 %.

## Reproduce

On a node with 32 free physical cores (the recorded runs: process times of 4 s for SSA at 4 km to 10 min for BP at
0.5 km).

| step | command | output | cost |
|---|---|---|---|
| 1 environments | `./setup.sh` | `env/` (PISM), `tools-env/` (compiler, SWIG), from the package lists in `environment/` | once; download |
| 2 adapter | `./build_bp_front_extension.sh` | `bp_front_extension/_bp_front_boundary.so` | a minute |
| 3 runs | `source pism_env.sh && ./run.sh` | `results/<case>/` (`summary.json`, `audit.json`, `geometry.nc`, the velocity fields) | ~20 min |

The shipped `results/<case>/` hold `summary.json` (every solve: time, Newton and Krylov iterations, memory,
convergence), `attempt.json` (the command that ran, with the authors' paths) and `audit.json` of the eight recorded
cases, without the velocity fields. `pism_per_solve.csv` lists every solve; `validation_*.json` and
`analytic_checks.json` record the tolerance, vertical-refinement and analytic checks (uniform sliding, free shelf)
quoted above.

## Layout

```
pism_diag.py          SSA driver (PISM through its Python bindings; one process per rank)
bp_diag.py            Blatter-Pattyn driver, with the front adapter
bp_front_extension/   FrontMatched.hh, bp_front_boundary.i: the subclass of PISM's Blatter solver (README inside)
build_bp_front_extension.sh, setup.sh, pism_env.sh, environment/   build and environments
run.sh                the eight cases
results/              per-case records, the two tables, velocity differences, validation and resolved settings
```
