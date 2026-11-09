# Cost of the reference solver: scaling and comparison with PISM (paper Sect. 5.4, App. D, Fig. 7, Table D1)

Cold solves (from zero velocity) of the Newton–CG solver with the identity mapping on the Pine Island geometry at
t = 0, at 4, 2, 1 and 0.5 km, for the MOLHO and SSA vertical bases, in single and double precision, on one RTX 4090;
the effect of the batching of the Hessian probes (`probe_batch`); and, in [`pism/`](pism/README.md), the same solves
with PISM 2.3.1 on 32 CPU cores (Fig. 7a–b, Table D1).

## Set-up

**Geometry.** The Pine Island state at t = 0 of the application runs (`../../glaciers/runs/data/glacier_pig/start_1km.nc`:
1 km, 469 × 564 nodes, calibrated friction, quasi-rigid band), resampled by `resample_pig.py` to 4 and 2 km (nested
subsampling; where it breaks the quasi-rigid band the node becomes band) and to 500 m (class-aware bilinear
interpolation of domain ice, band, wall, open water and dry land; surface rebuilt from flotation; masks recomputed).
Every grid passed an audit (`data/audit.json`): no trench at the band/wall boundary, no domain ice touching a wall,
ice area and volume within 0.1 % of 1 km. Grids: 118 × 142, 235 × 283, 469 × 564 and 937 × 1129 nodes; DOFs =
2 N_z N_x N_y over the full grid (N_z = 2 for MOLHO, 1 for SSA), 56 % of them on ice.

**Solver** (the "fast" settings of the application runs, `run_config.yaml`): Newton–CG with CG tolerance 10⁻²;
stop when the RMS change of the velocity stays ≤ 1 m yr⁻¹ for 3 consecutive iterations, at most 200 Newton
iterations; adaptive damping. The banded Hessian is extracted from graph-coloured Hessian–vector products, MOLHO
with `probe_batch: 9`, SSA with its compact assembly (`probe_batch: 0`). Preconditioners: the barotropic geometric
multigrid V-cycle with a point-block Jacobi smoother for MOLHO; point-block Jacobi for SSA.

**Timing.** Each case makes one call that traces and compiles (40–115 s, in the CSV but not in the figure), then 5
timed cold solves; the figure shows their median and range. One job per GPU, the GPU at or below 45 °C at the start
of each case (`run_queue.sh`), TensorFlow 2.17. The spread of the 5 solves is ≤ 4 % of the median. Warm solves
(from the previous velocity) always stop at the minimum of 4 Newton iterations, so the cost of a time step is about
the number of Newton iterations per step (≈ 9 at 1 km in the Pine Island transient) times the time per Newton
iteration (`time_per_newton_s`).

**Accuracy.** Every timed solve is compared with a float64 reference on the same grid (stop at 0.1 m yr⁻¹, CG
tolerance 10⁻⁴; continuing it to 0.01 m yr⁻¹ and 10⁻⁶ changes it by < 10⁻⁶): the relative RMSE of the speed over all
ice is ≤ 3 × 10⁻⁴ in float32 and ≤ 3 × 10⁻⁵ in float64. The 500 m MOLHO cases run out of memory at `probe_batch` 9,
in both precisions (also the reference); see below.

## Results (`results/scaling.csv`)

| Grid | MOLHO f64 (s) | MOLHO f32 (s) | SSA f64 (s) | SSA f32 (s) | MOLHO f64 / f32 (GiB) | SSA f64 / f32 (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| 4 km | 1.92 | 1.14 | 0.92 | 0.94 | 0.87 / 0.51 | 0.05 / 0.02 |
| 2 km | 4.38 | 1.75 | 0.99 | 0.98 | 3.3 / 1.7 | 0.20 / 0.10 |
| 1 km | 14.6 | 3.99 | 1.76 | 1.13 | 12.7 / 6.4 | 0.81 / 0.42 |
| 500 m | out of memory | out of memory | 7.68 | 2.99 | — | 2.6 / 1.35 |

Every valid solve takes 9 Newton iterations (10 for MOLHO float32 at 2 and 1 km). CG iterations per Newton iteration:
6–10 for MOLHO (multigrid) at every size; 58, 98–104, 204 and 400 for SSA (block Jacobi), doubling at each halving
of the grid spacing. A fixed cost of ≈ 0.11 s per Newton iteration dominates up to ~10⁵ DOFs; on the last doubling
the time grows as DOFs^0.87 (MOLHO f64) and DOFs^1.06 (SSA f64). float64 is 1.7×, 2.5× and 3.7× slower than
float32 for MOLHO at 4, 2 and 1 km, and up to 2.6× for SSA at 500 m. Memory grows linearly with the DOFs: ~6 GiB
per million DOFs for MOLHO in float32 at `probe_batch` 9 (mostly probe buffers), ~0.6 GiB for SSA.

## Probe batching (`results/cases/probe_*`, `../figure/data/probe_batch.csv`)

The bilinear stencil couples each node with its 8 neighbours, so 9 colours separate the columns of the Hessian;
with the 4 velocity components of MOLHO that makes 36 probes, independent of the grid size. `probe_batch` sets how
many probes go into one automatic-differentiation call (0: 36 separate calls; 9: four calls of nine); each batch
copies the state once per probe, so memory grows with the batch, and the solution is the same up to rounding. The
paper uses 9 in every MOLHO Newton–CG run (chosen for speed on a 70k-node grid, where it halved the time of the
Hessian extraction). On Pine Island at 1 km, batches of 3 or 9 make a Newton iteration ~10 % faster than 1 and ~20 %
faster than 0, for 4× the memory of 1 (6.4 against 1.6 GiB in float32); at 500 m (1.06 M nodes), 0, 1 and 3 are
equally fast and only 0 and 1 fit in both precisions (12 s in float32 and 58 s in float64 per cold solve).

## Software and hardware

- IGM `dev` (see `../../README.md`).
- One NVIDIA RTX 4090 (24 GiB, driver 580.178.04), Python 3.10, TensorFlow 2.17; `pip_freeze` of the environment
  recorded in `results/cases/*/summary.json` (`environment`).

## Reproduce

Run from this folder, in an environment with IGM (`PY=python-with-igm ./run_queue.sh …`; `IGM_SRC` can point to an
IGM source tree).

| step | command | output | cost |
|---|---|---|---|
| 0 (optional) grids | `python resample_pig.py` | `data/{4km,2km,1km,500m}/initial.nc`, `data/audit.json` (shipped) | a minute, CPU |
| 1 timed cases | `./run_queue.sh GPU queue_main.txt` | `cases/<case>/`, `work/raw/<case>/` (raw velocities), `work/logs/` | ~1.5 h, 1 GPU |
| 2 references | `./run_queue.sh GPU queue_ref.txt` | `cases/ref_*`, `cases/refB_*` | ~1 h |
| 3 probe batching | `./run_queue.sh GPU queue_probe.txt` | `cases/probe_*` | ~30 min |
| 4 analysis | `python analyze.py` | `results/scaling.csv`, `results/reference_stability.csv`, `cases/*/summary.json` | a minute, CPU |
| 5 figure data | `python export_figure_data.py` | `../figure/data/*.csv` (shipped) | seconds |
| 6 PISM | see [`pism/README.md`](pism/README.md) | `pism/results/` | ~20 min, 32 cores |
| 7 figure | `python ../figure/make_figure.py` | `../figure/fig_07_solver_cost.{pdf,png}` | seconds |

`pig_diag_bench.py` is the benchmark driver: it initialises only IGM's ice-flow module and calls its solver as a time
step does (`--mode timing`: compile call plus N cold solves; `--mode reference`: tight float64 solves). `analyze.py`
needs the raw velocity fields of a rerun (`work/raw/`, about 1.2 GB, not shipped); `results/` holds its outputs for
the paper's runs, with the per-case summaries (`results/cases/<case>/summary.json`, `summary_raw.json`,
`repetitions_raw.csv`). `export_figure_data.py` wrote `../figure/data/` from the authors' study folder and is kept as
provenance.

## Layout

```
pig_diag_bench.py   benchmark driver
run_config.yaml     the resolved application configuration the driver takes the solver settings from
queue_*.txt         the cases (name, mode, basis, precision, grid, extra driver arguments); queue_main.txt ends with the
                    0.5 km MOLHO solves at probe_batch 1 of Fig. 7a
run_queue.sh        runs a queue on one GPU, with thermal control
resample_pig.py     1-km state -> data/<res>/initial.nc
data/               the four grids and their audit
analyze.py          cases -> results/scaling.csv (times, iterations, memory, accuracy against the references)
results/            scaling.csv, reference_stability.csv, cases/<case>/summary*.json and repetitions_raw.csv
export_figure_data.py   provenance: how ../figure/data was made
pism/               the comparison with PISM (drivers, front adapter, environments, run.sh, results)
```
