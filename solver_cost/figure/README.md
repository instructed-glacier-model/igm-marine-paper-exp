# Solver-cost figure (Fig. 7)

```bash
python make_figure.py      # seconds; needs numpy, pandas, matplotlib. GMD_COST_XAXIS=spacing puts the grid spacing on x
GMD_FIG_LAYOUT=wide python make_figure.py   # fig_07_solver_cost_wide: panels side by side, the layout of the single-column manuscript
```

Writes `fig_07_solver_cost.{pdf,png}` (Fig. 7 of the two-column version; `fig_07_solver_cost_wide.pdf`, with the three panels side by side, is Fig. 7 of the single-column manuscript): the cost of a cold velocity solve on the
Pine Island geometry at 4, 2, 1 and 0.5 km, against the number of velocity DOFs. (a) PISM's Blatter–Pattyn solver
with two vertical levels on 32 CPU cores against IGM's MOLHO basis on one RTX 4090; (b) SSA in both codes; (c) peak
GPU memory of the IGM solves. PISM in orange; IGM MOLHO in ink, IGM SSA in blue; float64 solid with filled circles,
float32 dashed with hollow circles.

`data/`:

| file | content | panel |
|---|---|---|
| `cpu_gpu_comparison.csv` | the eight cases of the comparison: PISM median time on 32 cores, IGM median times in float32 and float64 (measured in the same campaign; 0.5 km MOLHO with `probe_batch` 1), velocity unknowns (`../runs/pism/`) | (a), (b) |
| `solver_scaling.csv` | the 16 cases of the scaling study (grid, basis, precision; `dofs`, `dofs_active_ice`; `cold_median_s`, `cold_min_s`, `cold_max_s` over 5 solves; `compile_call_s`; `newton_its`, `cg_its_per_newton`, `time_per_newton_s`; `peak_gpu_gib`; `rel_rmse_vs_float64_ref`; `status`, `plotted`); written by `../runs/export_figure_data.py` from `../runs/results/` | (c) |
| `pism_case_summary.csv` | every PISM case: timings, Newton and Krylov iterations, memory, convergence (Table D1) | not drawn |
| `probe_batch.csv` | MOLHO cold solves at `probe_batch` 0, 1, 3 and 9, at 1 km and 500 m (Sect. 5.4) | not drawn |
| `paper_case_grids.csv` | grid sizes and solver settings of the paper's test cases | not drawn |
