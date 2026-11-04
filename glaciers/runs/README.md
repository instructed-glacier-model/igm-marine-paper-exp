# Three glaciers: reference and neural-network runs (paper Sect. 5; Figs. 4–6 and 8; Tables 1, 2, C1)

Sermeq Kujalleq (Greenland; 150 m grid, 25 yr of ocean warming and prescribed front retreat), Pine Island Glacier
(Antarctica; 1 km, 100 yr of ocean warming) and Mendenhall Glacier (Alaska; 50 m, 25 yr of rising equilibrium-line
altitude). On each glacier, one **reference run** (identity mapping solved with Newton–CG) and four
**neural-network runs** (hybrid CNN + FNO trained online with SS-eSOAP; retraining budgets N = 5, 10, 20 and 50
iterations after the first year): 15 runs. The set-ups are summarised in Tables 1 and C1 of the paper and detailed
in [`how_the_results_were_obtained.md`](how_the_results_were_obtained.md).

| Glacier | Reference run | Neural-network runs (N = 5, 10, 20, 50) | Experiment files |
|---|---|---|---|
| Sermeq Kujalleq | `glacier_jak_P1_30_exact_fast_mt5_25yr` | `j25_n25_{N}_mt5` | `glacier_jak_P1_30_exact_fast`, `glacier_jak_P1_30_T2_init_stop` |
| Pine Island Glacier | `glacier_pig_exact_fast_mt5_100yr` | `pig_n50_{N}_mt5_100yr` | `glacier_pig_exact_fast_frames`, `glacier_pig_W1_T2_init_stop` |
| Mendenhall Glacier | `glacier_men_exact_fast_mt5_25yr` | `m25_n50_{N}_mt5` | `glacier_men_exact_fast_frames`, `glacier_men_E1_T2_init_stop` |

On a glacier, the runs differ only by the ice-flow mapping: same input state, physics, forcing and time stepping.
The retraining budget N and the run length are command-line overrides of the experiment file (`run_all.sh`).

## Data (`data/`)

The prepared input of each glacier (`start_*.nc`: geometry, calibrated friction, surface mass balance, water
level, observed speed), its flowline, and Pine Island's melt and flux corrections; [`data/README.md`](data/README.md)
gives their sources and how they were built (Appendix C of the paper).

## Software and hardware

- IGM `dev` (see `../../README.md`).
- One NVIDIA RTX 4090 (24 GiB) per run, driver 580.178.04; Python 3.10, TensorFlow 2.17.
- User modules in `user/`: `iceflow_geometry` (saves the geometry seen by the ice-flow solve), `save_times` (the
  saved frames: 0, 0.01, 0.05, 0.1, 0.25, 0.5 yr, then every year), `thin_ice` (removes ice thinner than 1 cm),
  `tf_correction` and `flux_correction` (Pine Island's static corrections), and the output `marine_ts` (time series
  of the grounding-line and front positions along the flowline, volume above flotation, areas, time step, wall
  time).

## Reproduce

Run from this folder, in an environment with IGM, one GPU per run.

| step | command | output | cost |
|---|---|---|---|
| 1 runs | `./run_all.sh` (or one of its 15 commands) | `outputs/<run>/` (`output.nc`, `marine_ts.csv`, `.hydra/config.yaml`) | reference runs 435, 136 and 7 min; network runs 4–134 min (one RTX 4090 each) |
| 2 Fig. 8 runs | `geometry_response/run.sh` | `outputs/w_*`, `outputs/netreplay_exact_*`, `outputs/replay_*` | see [`geometry_response/README.md`](geometry_response/README.md) |
| 3 figure data | `python export_figure_data.py` | `../figure/data/` (shipped) | minutes, CPU |
| 4 figures | `python ../figure/make_figures.py` | `../figure/fig_0{4,5,6,8}_*.{pdf,png}`, `../figure/data/summary.csv` | ~2 min, CPU |

`configs/<run>.yaml` is the resolved IGM configuration of each run, the authoritative record of every setting;
`configs/<run>_provenance.txt` its command, host, GPU, IGM commit and the sha256 of its input and user code.
`export_figure_data.py` is the script that wrote `../figure/data/` from the runs; it uses readers of the authors'
analysis project and is kept as provenance (its paths need adapting to rerun it).

## Layout

```
experiment/        the 6 experiment files (Hydra); the run-specific settings are overrides in run_all.sh
user/              user modules (code/ and conf/)
data/              inputs (README.md: sources and preparation)
configs/           resolved configuration and provenance of each of the 15 runs
run_all.sh         the 15 run commands
geometry_response/ the extra runs and the script of the error split of Sect. 5.5 (Fig. 8)
export_figure_data.py          provenance: how ../figure/data was exported from the runs
how_the_results_were_obtained.md  set-ups, solvers, training, metrics, timings and the error split, in detail
```
