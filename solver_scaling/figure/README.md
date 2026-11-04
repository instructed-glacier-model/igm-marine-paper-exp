# Solver-scaling figure (Fig. 7)

```bash
python make_figure.py      # seconds; needs numpy, pandas, matplotlib. GMD_SCALING_LAYOUT=wide for the side-by-side layout
```

Writes `fig_07_solver_scaling.{pdf,png}` (the PDF is Fig. 7 of the paper): (a) computing time of a cold solve and
(b) peak GPU memory against the number of velocity DOFs. MOLHO in ink, SSA in blue; float64 solid with filled
circles, float32 dashed with hollow circles. Only valid solves are drawn (`plotted`): converged and within 1 % of
the float64 reference.

`data/`, written by `../runs/export_figure_data.py` from `../runs/results/`:

| file | content |
|---|---|
| `solver_scaling.csv` | the 16 cases (grid, basis, precision; `dofs`, `dofs_active_ice`; `cold_median_s`, `cold_min_s`, `cold_max_s` over 5 solves; `compile_call_s`; `newton_its`, `cg_its_per_newton`, `time_per_newton_s`; `peak_gpu_gib`; `rel_rmse_vs_float64_ref`; `status`, `plotted`) |
| `probe_batch.csv` | MOLHO cold solves at `probe_batch` 0, 1, 3 and 9, at 1 km and 500 m (not drawn; Sect. 5.4) |
| `paper_case_grids.csv` | grid sizes and solver settings of the paper's test cases (not drawn) |
