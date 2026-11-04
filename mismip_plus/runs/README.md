# MISMIP+: grounding-line migration (paper Sect. 4.2, Fig. 3)

The MISMIP+ protocol (Asay-Davis et al., 2016, doi:[10.5194/gmd-9-2471-2016](https://doi.org/10.5194/gmd-9-2471-2016))
run with IGM on a 1-km grid: the Ice0 steady state and control, then Ice1r / Ice1rr / Ice1ra (melt concentrated
near the grounding line) and Ice2r / Ice2rr / Ice2ra (100 m yr⁻¹ beyond x = 480 km), compared with the 24 models of
the main subset of Cornford et al. (2020, doi:[10.5194/tc-14-2283-2020](https://doi.org/10.5194/tc-14-2283-2020)).

| | protocol | IGM setting |
|---|---|---|
| domain | 640 × 80 km, x from the divide to the front | 641 × 81 nodes at 1 km (nodes on the walls and the centerline) |
| bed | Asay-Davis et al. (2016), Eq. 1 | `prep/common.py` |
| densities, g | 918, 1028 kg m⁻³, 9.81 m s⁻² | `ice_density`, `water_density`, `thk.ratio_density` 0.892996 |
| rheology | Glen n = 3; A = 2.52 × 10⁻¹⁷ Pa⁻³ yr⁻¹ (the value of the PISM submissions) | `arrhenius` 25.2455 MPa⁻³ yr⁻¹; the Ice0 grounding line then crosses the centerline at 446 km, the ensemble mean |
| friction | Weertman, m = 1/3, τ_ref = 10 kPa at u_ref = 1 m yr⁻¹, none under floating ice | `law: weertman`, no sub-grid treatment at the grounding line |
| stress balance | any | Blatter–Pattyn, MOLHO (N_z = 2, Q1 elements) |
| ice-flow mapping | — | neural-network mapping (hybrid CNN + FNO, online training with SS-eSOAP, float32) for the runs of Fig. 3; identity mapping solved with Newton–CG (float64) for the comparison of the text |
| accumulation | 0.3 m yr⁻¹ | `smb simple` |
| boundaries | no slip at x = 0, free slip at y = ±40 km, calving front at x = 640 km | Dirichlet velocity at x = 0; `periodic_ns` with the mirror-symmetric bed; floating-front stress at x = 640 km, ice beyond it removed |
| Ice1 melt | Eq. 7 of the protocol | `bmb` `prescribed.form: mismip_plus` |
| Ice2 melt | 100 m yr⁻¹ for x > 480 km | `bmb` `prescribed.form: field` (field of the input) |
| melt at the grounding line | not prescribed | melt on fully floating cells only (`nmp`) |
| thin floating ice | — | floating ice thinner than 10 m removed (`thin_ice_calving`), which makes the melted region of Ice2 an ice front |

## Runs

| Run | What | Input |
|---|---|---|
| `ice0_nn_1km` | Ice0 spin-up from a 100-m slab, 10.8 kyr, until steady (`steady_stop`) | `mismip_plus_1km_slab100.nc` |
| `ice0_1km_nn`, `ice1r_1km_nn`, `ice2r_1km_nn` | Ice0 control, Ice1r and Ice2r, 0–100 yr (Fig. 3) | `restart/ice0_1km.nc`, the end state of the spin-up |
| `ice1rr_1km_nn`, `ice1ra_1km_nn`, `ice2rr_1km_nn`, `ice2ra_1km_nn` | melt maintained (rr) or stopped (ra), 100–200 yr (Fig. 3) | `restart/ice{1,2}r_1km_nn_t100.nc`, the 100-yr state of the r run |
| `ice0_1km`, `ice{1,2}{r,rr,ra}_1km` | the same seven experiments with the identity mapping solved with Newton–CG (float64): the grounding line differs from the network runs by less than 2 km at any time (text) | the same Ice0 state; `restart/ice{1,2}r_1km_t100.nc` |

`configs/<run>.yaml` is the resolved configuration recorded for each run. The experiment files were regenerated
after the Ice0 and Ice1 runs with `thin_ice_calving` added to every experiment; it acts in Ice2 only (the Ice0 and
Ice1 runs had no floating ice thinner than 10 m), so the shipped `experiment/*.yaml` reproduce every run.

## Data (`data/`) and the ensemble

- `mismip_plus_1km.nc`, `mismip_plus_1km_slab100.nc` (`prep/make_inputs.py`): bed, sea level, Ice2 melt field, and
  the initial thickness (the protocol's profile, or a 100-m slab).
- `restart/`: the Ice0 state and the 100-yr states of the paper's runs (`make_restart.py`).
- The ensemble of Cornford et al. (2020) is not shipped: `python prep/fetch_ensemble.py` downloads it (20.7 MB,
  md5-checked) from Zenodo, doi:[10.5281/zenodo.3611936](https://doi.org/10.5281/zenodo.3611936) (CC BY 4.0), into
  `raw/ensemble/`. `ensemble.py` reads the submissions and selects the main subset and its Weertman models as in
  Cornford et al. (2020, Sect. 4.2). The envelopes drawn in Fig. 3 are shipped in `../figure/data/`.

## Software and hardware

- IGM `dev` (see `../../README.md`).
- One NVIDIA RTX 4090 (24 GiB) per run, driver 535.161.07.
- User modules in `user/`: `mismip_ts` (output: volume, VAF, grounded area, grounding-line positions, wall clock),
  `steady_stop` (stops the spin-up once steady), `thin_ice_calving`.

## Reproduce

Run from this folder, in an environment with IGM.

| step | command | output | cost |
|---|---|---|---|
| 0 (optional) inputs and experiments | `python prep/make_inputs.py --slab 100 && python prep/make_inputs.py && python prep/make_experiments.py` | `data/*.nc`, `experiment/*.yaml` (shipped; the generator also writes other grids and variants) | seconds |
| 1 network runs | `./run.sh` (block 1) | `outputs/*_1km_nn/`, `data/restart/` | spin-up 7 min, the seven experiments 28 min (1 GPU) |
| 2 Newton–CG runs | `./run.sh` (block 2) | `outputs/*_1km/` | ~2.5 h |
| 3 ensemble | `python prep/fetch_ensemble.py` | `raw/ensemble/` | 20.7 MB download |
| 4 figure data | `python make_figure_data.py` | `../figure/data/*.csv` (shipped) | seconds |
| 5 figure | `python ../figure/make_figure.py` | `../figure/fig_03_mismip_timeseries.{pdf,png}` | seconds |

Wall times of the paper's runs (launch to exit, including start-up and the initial training of the network): with the
network, 3.9 min (Ice0 control), 4.1 / 4.0 / 3.2 min (Ice1r / rr / ra), 4.6 / 4.2 / 3.7 min (Ice2); with Newton–CG,
20.8, 21.2 / 21.8 / 15.1 and 33.4 / 29.6 / 26.4 min.

## Layout

```
prep/              common.py (protocol constants, bed), make_inputs.py, make_experiments.py, fetch_ensemble.py, methods/
experiment/        the 15 run configurations (generated; self-contained)
configs/           the resolved configuration recorded for each run
user/              user modules (code/ and conf/): mismip_ts, steady_stop, thin_ice_calving
data/              inputs and restart files
run.sh             the runs in sequence
make_restart.py    a frame of a run -> data/restart/<name>.nc
check_steady.py    steady-state criteria of the spin-up on mismip_ts.csv
ensemble.py        reader of the Cornford et al. (2020) ensemble and of its subsets
make_figure_data.py  runs + ensemble -> ../figure/data/*.csv
```
