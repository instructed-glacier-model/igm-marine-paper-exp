# How the results were obtained

> **Terminology.** These notes were written before the final wording of the paper. Here, "Newton–CG" is the
> paper's *reference run* (identity mapping solved with Newton–CG), and "the emulator" is the paper's
> *neural-network run* (network mapping trained online). Section numbers of the paper are not used.

The settings below are taken from the resolved run configurations in `configs/` (`RUN.yaml`). Hardware, code and
input provenance come from `configs/RUN_provenance.txt`. Wherever this file and a config disagree, the config is
right.

Three names recur:
- **Newton–CG**: the reference solver.
- **The emulator**: the online-trained network.
- **N**: the number of training iterations per retraining after the first year.

## 1. The 15 runs

| Glacier | Newton–CG run | Emulator runs (N = 5, 10, 20, 50) | Length |
|---|---|---|---|
| Sermeq Kujalleq (Jakobshavn Isbræ), Greenland | `glacier_jak_P1_30_exact_fast_mt5_25yr` | `j25_n25_{N}_mt5` | 25 yr |
| Pine Island Glacier, Antarctica | `glacier_pig_exact_fast_mt5_100yr` | `pig_n50_{N}_mt5_100yr` | 100 yr |
| Mendenhall Glacier, Alaska | `glacier_men_exact_fast_mt5_25yr` | `m25_n50_{N}_mt5` | 25 yr |

- **How each pair differs.** On a given glacier, the Newton–CG run and the emulator runs share everything except
  the ice-flow solver: same input state, physics, forcing and time stepping. Each run evolves its own geometry, so
  the comparison is between two trajectories.
- **Saved frames:** 0, 0.01, 0.05, 0.1, 0.25 and 0.5 yr, then every year. That gives 31 frames on jak and men and
  106 on PIG. All errors are evaluated on these frames, after the runs have finished; no run computes an error while
  it runs.
- **Hardware and software:**
  - every run had one NVIDIA GeForce RTX 4090 (24 GB) to itself, on nodes 28–32 of the same cluster (`costs.csv`:
    `node`);
  - Python 3.10 and TensorFlow 2.17;
  - IGM at commit `058558bc` (development branch).
- **Two code states.** 8 of the 15 runs ran with an uncommitted four-file change that adds the SS-eSOAP option
  `relative_damping = 1e-6`. That change was later committed unchanged in IGM `48e0b713`. `costs.csv` (`igm_code`)
  lists which run used which state:
  - **without the change:** jak Newton–CG and N = 5, 10, 20; PIG N = 5, 10, 20;
  - **with it:** jak N = 50, PIG Newton–CG and N = 50, and all five men runs.
  - **The option:** a ridge, relative to the factor's scale, on the SS-eSOAP Kronecker factors before their
    float32 eigendecomposition. It prevents a rare cuSOLVER convergence failure, which had stopped one earlier
    N = 1 run.
  - **Effect:** none on the Newton–CG runs, which do not use SS-eSOAP. On the emulator it changes training at
    rounding level, like another random seed. Controls with and without the change on jak and men agree to
    rounding level (grounding line ±0.05 km, VAF ±2 % of its change).
  - **Visible consequence:** the automatic stop of the initial training (Sect. 5) stops at a different iteration
    for N = 50 than for N = 5–20 on jak and PIG.

## 2. Glaciers and forcing

| | Sermeq Kujalleq | Pine Island Glacier | Mendenhall Glacier |
|---|---|---|---|
| Grid | 150 m, 667 × 401, EPSG:3413 | 1 km, 469 × 564, EPSG:3031 | 50 m, 354 × 409, UTM 8N |
| Data | BedMachine Greenland v6.6; ITS_LIVE v2 (120 m); RACMO2.3p2 1995–2014 SMB | BedMachine Antarctica v4.1; ITS_LIVE v2; RACMO2.3p2 ANT27 1995–2014 SMB; domain of De Rydt et al. (2021); seabed near the grounding line deepened after Bett et al. (2024) | OGGM 1.6 on Copernicus GLO-30; Farinotti et al. (2019) consensus thickness; ITS_LIVE composite (via OGGM); RGI60-01.00709 |
| Ice rheology | Glen n = 3, uniform A = 33.1 MPa⁻³ yr⁻¹ (Cuffey and Paterson 2010 at −10 °C, × enhancement 3) | A = 11.0 MPa⁻³ yr⁻¹ (−10 °C) | A = 75.7 MPa⁻³ yr⁻¹ (temperate, 0 °C) |
| Water | ocean, ρ_w = 1023 kg m⁻³ | ocean, ρ_w = 1027 kg m⁻³ | proglacial lake at 15 m, ρ_w = 1000 kg m⁻³ |
| Forcing ("scenario") | **P1_30:** ocean warming δT from 0 to +2 K linearly over 30 yr (+1.67 K at 25 yr), through sub-shelf melt; **prescribed front retreat** of 500 m yr⁻¹ (calving law `ice_speed`, front velocity −500 m yr⁻¹) | **W1:** ocean warming δT from 0 to +1 K over 5 yr, then held | **E1:** equilibrium-line altitude rising from 1236 to 1416 m over 30 yr (+6 m yr⁻¹; 1386 m at 25 yr) |
| Sub-shelf melt | quadratic local law, γ0 = 65 317 m yr⁻¹, fitted to a tongue-mean melt of 230 m yr⁻¹ (Motyka et al. 2011); no melt at partly grounded cells (nmp); Disko Bay-like profile with 2 °C Atlantic water below 300 m | quadratic local law (ISMIP6 form), γ0 = 49 500 m yr⁻¹; melt at partly grounded cells scaled by their floating fraction (pmp); refreezing allowed; a static per-cell thermal-forcing correction sets the initial melt to the balance melt minus the observed thinning rate (Smith et al. 2020) | none |
| SMB | static (input) | static (input), plus a static flux correction on grounded ice (after Hill et al. 2023) | IGM `simple` (gradients 0.009 / 0.005 m w.e. m⁻¹, max accumulation 2 m w.e. yr⁻¹), updated yearly |
| Calving front | moves (sub-grid front), retreat prescribed | fixed (`thk.front.fixed`) | moves (sub-grid front); water-depth calving c = k d, k = 2 yr⁻¹ |
| Domain edges | ice crosses the box edges (Dirichlet thickness) | zero thickness | zero thickness |

- **Initial state.**
  - τ_ref, the Weertman sliding coefficient, was calibrated per cell against the observed surface speed on the
    input geometry. This is a fixed-point relaxation, with one cold Newton–CG steady solve per update. Floating ice
    is not calibrated.
  - Each glacier was then relaxed with present-day forcing and fixed fronts: jak for 2 yr, PIG and men for 5 yr.
  - All 15 runs start from the end of that relaxation, at t = 0.
- **Pine Island's quasi-rigid band.** A narrow band of the domain carries a very large τ_ref (it is close to rigid).
  It is left out of every error and drawn as land on the maps (`keep = 0` in `data/maps_pig.nc`).
- **PIG forcing after 50 yr.** The anomaly table ends at 50 yr and is held constant after it, so +1 K applies from
  5 to 100 yr.
- **Ice thinner than 0.01 m is removed** on jak and PIG (user process `thin_ice`). Without it, cells with vanishing
  ice have undetermined velocities that collapse the CFL time step.

## 3. Common physics and time stepping

- **Ice flow:**
  - MOLHO (two vertical basis functions), Q1 elements, single precision (float32);
  - Weertman sliding with exponent 3 on grounded ice only, and floating ice frictionless;
  - energy components: viscosity, gravity, sliding, floating (ocean or lake back-pressure at the front).
- **Geometry:**
  - explicit upwind finite volumes with a superbee limiter;
  - sub-grid calving front (first order, flux threshold, minimum front thickness 5 m);
  - grounded where the height above flotation is positive.
- **Time step:** adaptive CFL 0.3, capped at 0.1 yr.
  - The steps then come out at: 8400–8560 steps in 25 yr on jak (mean 0.003 yr, set by the fast calving front);
    2050–2075 in 100 yr on PIG (0.05 yr); 266–267 in 25 yr on men (0.09 yr).
  - The step counts of the emulator runs differ from Newton–CG's by less than 2 %.
- **Ice-flow solve:** every step. The Newton–CG run solves at every step. The emulator is evaluated at every step
  and retrained at every second step.

## 4. Newton–CG (the reference)

- **Method:** identity mapping (the velocity degrees of freedom themselves), minimised by the paper's matrix-free
  Newton–CG.
  - The Hessian is extracted as a banded stencil from graph-coloured Hessian–vector products.
  - The preconditioner is a barotropic multigrid V-cycle.
  - Armijo line search.
  - Adaptive damping.
- **Per time step:**
  - up to 20 Newton iterations, warm-started from the previous step's velocity;
  - the CG tolerance is 0.01 (`cg_tol`), with at most 4000 CG iterations;
  - the solve stops when the RMS change of the velocity between Newton iterations stays below 1 m yr⁻¹ for 3
    iterations in a row.
- **Initial solve:** from zero velocity, up to 200 Newton iterations. Once a year (model time) the solve restarts
  from zero velocity instead of the previous step's velocity (`cold_restart_interval: 1`).
- **Wall time:** 434.8 min (jak), 135.6 min (PIG) and 7.4 min (men), on one RTX 4090. The time stepping alone
  takes about 3.0, 3.9 and 1.3 s per step.

## 5. The emulator

- **Network:** a hybrid of a CNN and a Fourier neural operator (IGM architecture `hybrid_dahunet_fno`).
  - **CNN:** 12 layers, 16 filters, 3 × 3 kernels, residual.
  - **FNO:** 2 layers, width 32, 8 × 8 Fourier modes.
  - **Inputs:** thickness, surface elevation, A, τ_ref, the grid spacing and the water level, normalised.
  - **Physics-based input features:** the surface gradients, plus the SIA and Weertman velocity estimates
    (Gregov et al. 2026c).
  - **Seed:** 0 in every run.
- **Training is online and uses no data.** The loss is the same discretised energy functional that Newton–CG
  minimises, evaluated on the current geometry (Jouvet and Cordonnier 2023; Sect. "Network mapping" of the paper).
  - The optimiser is SS-eSOAP: learning rate 3 × 10⁻⁴, β = (0.9, 0.95), damping 10⁻⁸ (plus the relative damping of
    Sect. 1 on 8 runs).
- **Initial training with an automatic stop** (IGM option `init_stop`):
  - At t = 0, one direct Newton–CG solve of the initial state is made: the same solver and halt as the reference
    run, from zero velocity, up to 200 Newton iterations.
  - The network then trains for up to 4000 iterations. Every 50 iterations its surface speed s is compared with
    the direct one s_ref, as the median of |s − s_ref| / (s_ref + 10 m yr⁻¹) separately over grounded and over
    floating ice.
  - Training stops when both errors are below their targets, 1 % (grounded) and 2.5 % (floating). It also stops
    when the error has not improved by 5 % over 500 iterations, and in any case at 4000 iterations.
  - Where it stopped (`costs.csv`):

    | | Stop | Iterations | Grounded / floating error |
    |---|---|---|---|
    | jak N = 5–20 | targets met | 2150 | 0.90 / 2.14 % |
    | jak N = 50 | targets met | 1500 | 0.97 / 2.48 % |
    | PIG N = 5–20 | plateau | 2700 | 1.61 / 5.91 % |
    | PIG N = 50 | plateau | 2450 | 1.43 / 3.82 % |
    | men | cap reached | 4000 | 0.50 / 3.30 % |

  - The direct solve is only used to stop the training. It does not set the run's initial velocity, which comes
    from the network.
- **Retraining:**
  - Retraining happens at every second time step (`retrain_freq = 2`).
  - In the first year, every retraining does N₁ iterations: 25 on jak, 50 on PIG and men. The ice-flow response
    to the start of the forcing and to the first front motion is fastest then.
  - After the first year, every retraining does N iterations, with N = 5, 10, 20 or 50: the variable studied
    (IGM option `nbit: [[0, N₁], [1, N]]`).
  - Each iteration is one SS-eSOAP update on the whole domain (one patch, no batching).
  - Between retrainings the network is only evaluated.

## 6. Error metrics (`data/errors.csv`)

All metrics are computed per frame, between the emulator run and the Newton–CG run at the same time.

- **Cells:** the ice in both runs (Pine Island's quasi-rigid band left out), with grounded and floating ice taken
  together.
- **Relative RMSE** `r_all` (%): ‖u − u_NCG‖₂ / ‖u_NCG‖₂ of the surface speed over those cells. It is the RMSE
  divided by the RMS speed of Newton–CG. Its grounded and floating parts are `r_gr` and `r_fl`; the RMSE itself
  (m yr⁻¹) is `rmse_all`, and the cell counts are `n_*`.
- **Mean relative RMSE** (Fig. 2e and the tables): the time mean of `r_all`, uniform in time (trapezoid between
  frames, so the dense early frames are not over-weighted).
- **Thickness:** median and RMSE of |Δh| (`h_med`, `h_rmse`).
- **Grounding-line and front positions:**
  - `s_gl` and `s_cf` are measured along a fixed flowline (`data/timeseries.csv`). Each is the downstream end of the
    last grounded (respectively ice-covered) stretch at least 2 km long, interpolated between flowline samples.
  - `d_gl` and `d_cf` (km) are the emulator's change minus Newton–CG's.
- **Volume above flotation:** `d_vaf` (km³) is the emulator's change minus Newton–CG's.
- **These are trajectory errors.** They contain:
  - the emulator's error on the geometry it sees;
  - the effect of its geometry having drifted from Newton–CG's, e.g. a grounding-line jump a few years earlier or
    later.

  Section 8 separates the two.
- **Fig. 3's relative difference** is (u − u_NCG) / (|u_NCG| + 10 m yr⁻¹). The 10 m yr⁻¹ floor stops slow ice from
  dominating, but slow marginal ice can still show 10–25 % for differences of a few m yr⁻¹.
- **A floored median** (`e_gr`, `e_fl`), the median of |Δu| / (|u_NCG| + 10 m yr⁻¹), is also in the table. It is
  the metric of the automatic stop, and is not used in the figures.

## 7. Computing time (`data/costs.csv`)

- **Wall time:** from the launch of the run to its exit, including Python and TensorFlow start-up, graph tracing
  and XLA compilation.
- **Split for the emulator runs (Fig. 2d):**
  - **start-up:** to the first output row, minus the next two parts;
  - **direct solve** and **initial training** of the automatic stop, as timed by IGM;
  - **time stepping:** the rest.
- **One-off cost:** start-up 0.4–0.6 min, direct solve 1.5–1.8 min and initial training 1.7–2.6 min, about 4.5 min
  in all. That is 1 % of the Newton–CG run on jak, 3 % on PIG and 61 % on men.
- **Time stepping per step,** for the emulator: about 0.14 s (N = 5) to 0.94 s (N = 50) on jak, 0.16–1.05 s on PIG
  and 0.18–0.78 s on men. Newton–CG takes 1.3–3.9 s.
- **Speed-up:** the Newton–CG wall time divided by the emulator's.
- **Variation between nodes:** a few percent (same GPU type).

## 8. Error split of Fig. 4 (after Rosier et al., egusphere-2026-5523, Sect. 3.3, Eq. 11)

**Notation:**
- R: the Newton–CG run's state at a frame; E: the emulator run's state at the same time;
- V: the mean surface speed over the cells with more than 1 m of ice in both states;
- V_net: the network of that frame; V_ex: Newton–CG.

**The split:**
- e_tot = V_net(E) − V_ex(R) = e_inst + e_geom;
- e_inst = V_net(R) − V_ex(R): the emulator's own error on Newton–CG's state;
- e_geom = V_net(E) − V_net(R): how the emulator's speed responds to the geometry difference;
- e_phys = V_ex(E) − V_ex(R): how Newton–CG's speed responds to the same difference.

If the emulator responded to geometry like the solver, then e_geom = e_phys, and the points of Fig. 4 would lie on
the 1:1 line.

**How each term was computed (N = 20 only):**
- **Saving the weights.** The emulator runs were rerun with a process that saves the network weights at every
  frame. The reruns are bitwise identical to the originals.
- **V_net(R).** Each frame's network was evaluated, without training, on the Newton–CG state of that frame. A
  self-check on the run's own states reproduces its saved speeds exactly.
- **V_ex(E) and V_ex(R).** These come from cold Newton–CG solves of every saved state of both runs: one IGM run per
  source run, each frame from zero velocity, with a tight tolerance (CG 10⁻⁴, halt 0.1 m yr⁻¹), float32.
  - Replaying the Newton–CG run as well means both terms use the same numerics, so e_phys = 0 at t = 0.
  - The replay reproduces single snapshot solves to ≤ 0.001 %, and float32 matches float64 to 0.05 % (relative
    L2).
  - On Pine Island, the float32 CG stalled on the nearly detached block (below), so the emulator run's frames
    from 57 yr were solved in float64.
- **Exclusion on Pine Island.** Only the emulator run forms a nearly detached floating block. On it, Newton–CG's
  solution on the emulator state runs away (up to ~10⁵ m yr⁻¹). Cells where V_ex(E) exceeds twice the maximum of
  V_ex(R) are left out of e_phys, about 95 cells per frame from about 50 yr.

**Results** (`data/geometry_response.csv`; time means of |e|, m yr⁻¹; corr is the correlation of e_geom and e_phys
over the frames):

| | V_ex(R) | \|e_tot\| | \|e_inst\| | \|e_geom\| | \|e_phys\| | corr |
|---|---:|---:|---:|---:|---:|---:|
| Sermeq Kujalleq | 608 | 6.65 (1.1 %) | 6.21 (1.0 %) | 1.15 | 4.32 | +0.90 |
| Pine Island | 332 | 3.89 (1.2 %) | 3.42 (1.0 %) | 0.95 | 2.96 | +0.30 |
| Mendenhall | 22.5 | 0.12 (0.6 %) | 0.15 (0.6 %) | 0.02 | 0.03 | +0.72 |

**Reading:**
- **On all three glaciers, the instantaneous error dominates** the total.
- **Online retraining keeps e_geom small:** the network is retrained on the geometry it actually sees.
- **On the marine glaciers the emulator under-reacts to geometry change.**
  - On Sermeq Kujalleq, e_geom follows e_phys closely in time (r = 0.90) but at about a quarter of its amplitude.
  - On Pine Island it is about a third, with a weak correlation.
  - So the emulator's speed responds too weakly when its geometry departs from Newton–CG's (e.g. during the
    grounding-line retreat).
- **Mendenhall:**
  - e_tot (0.12) is below e_inst (0.15): there is some cancellation between e_inst and e_geom;
  - this is the same structure as reported by Rosier et al.;
  - both terms are tiny.
- **Sermeq Kujalleq:**
  - e_inst is persistently negative (the network is about 1 % slow);
  - e_geom has the opposite sign for the first ~10 yr (slight cancellation), then the same sign during the
    grounding-line retreat;
  - so e_tot > e_inst on average.
- **Comparison with Rosier et al.:**
  - They report e_inst ≈ 1.2–1.9, e_geom ≈ 0.8–1.2 and e_tot ≈ 0.5–0.8 m yr⁻¹, with corr(e_geom, e_phys) ≈
    0.5–0.8: a total below the instantaneous error, i.e. cancellation. (These numbers were read from the preprint
    during this study; check them against the paper before quoting.)
  - Here, only Mendenhall shows cancellation.
- **The speed used:** this split uses the surface speed. Rosier et al. used the depth-averaged speed; switching was
  left open.

## 9. Findings worth knowing for the text

- **Accuracy:**
  - The mean relative RMSE is 3.3–4.1 % on Sermeq Kujalleq, 3.3–5.0 % on Pine Island and 0.9–1.5 % on Mendenhall.
  - After the first year it stays below 10 % on every glacier (`summary.csv`: `r_max_after_1yr`, at most 9.6 %).
  - More iterations help unevenly:
    - Pine Island and Mendenhall improve steadily from N = 5 to 50 (4.96 → 3.30 % and 1.53 → 0.91 %);
    - Sermeq Kujalleq stops improving after N = 10 (4.09, 3.25, 3.37, 3.38 %).
- **Cost:**
  - Speed-ups are 3–18× on Sermeq Kujalleq and 3–13× on Pine Island.
  - On Mendenhall the speed-up is 0.9–1.4×. The ~4.5 min one-off cost (start-up, direct solve, initial training) is
    61 % of the 7.4-min Newton–CG run, so the emulator doesn't pay off on a small, slow glacier with few time steps.
- **Grounding line:**
  - Grounding-line retreat by Newton–CG: 16.3 km (jak), 70 km (PIG), 1.3 km (men).
  - **jak:** Newton–CG's grounding line jumps 4.3 km at 23.9 yr. Only N = 5 has made that jump by 25 yr, hence
    ΔGL_end ≈ +4.3 km for N = 10–50 and +0.02 km for N = 5. Otherwise the trajectories overlap; N = 5 is at most
    0.23 km off.
  - **PIG:** the emulator runs reach the same final grounding line (±0.05 km). Mid-run they are up to 7 km apart,
    because the jumps happen at slightly different times.
  - **men:** the glacier leaves its lake after about 15 yr. Its flowline position then follows the land terminus.
- **VAF:** the final ΔVAF is within 3.1 % of Newton–CG's change on jak, 1.6 % on PIG and 0.1 % on men.
- **Sermeq Kujalleq's error peak at ~21 yr** (floating ice, every N) is geometry drift, not an emulator failure.
  - During a fast ungrounding (the shelf grows from 138 to 316 floating cells), the emulator's grounding line
    trails Newton–CG's by 0.05–0.14 km and its front by 0.1–0.3 km. The shelf speed is very sensitive to that.
  - Newton–CG solves on the emulator's own geometry at 19–23 yr put the emulator within 1.5 % of the solver there.
    The rest of the error is the drift, which closes again by 22–23 yr.
  - The hypothesis that grounding-line changes cause the error peaks was tested on all three glaciers and is not
    supported.
