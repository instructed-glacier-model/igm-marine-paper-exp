# Three glaciers figures (Figs. 4, 5, 6, 8, C1) and tables (1, 2, C1)

```bash
python make_figures.py     # ~2 min; needs numpy, scipy, pandas, matplotlib, xarray, netCDF4, cmcrameri
```

| Paper | Script | Output |
|---|---|---|
| Fig. 4 | `fig_gmd_glaciers.py` | `fig_04_glaciers_grounding_lines.{pdf,png}` |
| Fig. 5 | `fig_gmd_performance.py` | `fig_05_accuracy_and_cost.{pdf,png}`; also rewrites `data/summary.csv` |
| Fig. 6 | `fig_gmd_speed_difference.py` (`--abs` also draws the absolute difference, not in the paper) | `fig_06_speed_difference.{pdf,png}` |
| Fig. 8 | `fig_gmd_geometry_response.py` | `fig_08_geometry_response.{pdf,png}` |
| Fig. C1 | `fig_gmd_profiles.py` (geometry and speed along the flowlines of Fig. 4) | `fig_C1_flowline_profiles.{pdf,png}`; `_wide`: the single-column (preprint) layout |
| Tables 1 and 2 | `data/summary.csv`, `data/errors.csv`, `data/costs.csv`, `data/timeseries.csv` | — |
| Table C1 | `../runs/configs/<run>.yaml`, `data/costs.csv` (initial training and its stop) | — |

`common.py` holds the runs, the readers of `data/` and the style (7 pt, Nimbus Sans, one colour per N). Nothing
glaciological is computed when drawing: only masks, the grounding-line contour (height above flotation = 0) and
time means.

## Data (`data/`, 15 MB)

Written by `../runs/export_figure_data.py` from the 15 runs (and the runs of `../runs/geometry_response/`):

| file | content |
|---|---|
| `maps_<glacier>.nc` | grids of the reference run at t = 0 and at the end, and of the N = 20 network run at the end (thickness, bed, water level, surface speed), the domain mask (`keep`, without Pine Island's quasi-rigid band), the flowline, the densities |
| `timeseries.csv` | time series of every run (grounding-line and front positions along the flowline, volume above flotation, areas, time step, wall time), long format |
| `errors.csv` | per saved frame, the errors of each network run against the reference run (relative RMSE of the surface speed over all, grounded and floating ice; thickness; grounding-line, front and VAF differences) |
| `costs.csv` | per run, the wall time and its parts (start-up, direct solve, initial training, time stepping), the GPU and IGM code state, and the outcome of the automatic stop of the initial training |
| `geometry_response.csv` | the error split of Sect. 5.5 (`../runs/geometry_response/`) |
| `summary.csv` | the per-run summary of the tables, recomputed by `fig_gmd_performance.py` |
