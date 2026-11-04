# Inputs of the three glaciers

Each run reads one NetCDF file (`inputs.local.filename` in the experiment file, relative to this folder) and the
glacier's flowline; Pine Island also reads two correction fields.

| File | Content | sha256 |
|---|---|---|
| `glacier_jak/start_150m.nc` | Sermeq Kujalleq, 150 m, 667 × 401 (EPSG:3413) | `a6b440276ea71b87ae418e756f5e9db95ef61fa14fd2045fd5893bd4a149cab8` |
| `glacier_pig/start_1km.nc` | Pine Island Glacier, 1 km, 469 × 564 (EPSG:3031) | `fe30f34c7815bb8ab0bb5f3e16c77da0e132ff89ca94cd53ddd9d469f11324b3` |
| `glacier_men/start_50m.nc` | Mendenhall Glacier, 50 m, 354 × 409 (UTM 8N) | `8632db1967169f92b74a2d1817bcc012de8ec141d4ed50279f5eb46f1b6b8ce5` |
| `glacier_pig/tf_correction_obs_1km.nc` | static thermal-forcing correction (sets the initial melt to the balance melt minus the observed thinning) | `a50f2e54d441a4bdeccb69fc27b3bc9b2c0744e293ce5168c60f931a446a90ad` |
| `glacier_pig/flux_correction_obs_1km.nc` | static flux correction on grounded ice | `a024d37fc2f1fa13201dc70571dd960da9e14e6174642b8c28e54035bd817257` |
| `glacier_*/flowline.csv` | the flowline along which `marine_ts` measures the grounding-line and front positions | — |

Each `start_*.nc` holds the bed, thickness and surface, the calibrated Weertman coefficient `tau_ref`, the surface
mass balance, the water level, the observed surface speed, and the masks (ice, domain; Pine Island's quasi-rigid
band `bandmask`).

## How they were built (Appendix C of the paper)

1. **Raw data.** Sermeq Kujalleq: BedMachine Greenland v6 (Morlighem et al., 2017,
   doi:[10.1002/2017GL074954](https://doi.org/10.1002/2017GL074954)), ITS_LIVE v2 surface velocity (Gardner et al.,
   2022), RACMO2.3p2 1995–2014 surface mass balance (Noël et al., 2018,
   doi:[10.5194/tc-12-811-2018](https://doi.org/10.5194/tc-12-811-2018)). Pine Island: BedMachine Antarctica v4
   (Morlighem et al., 2020, doi:[10.1038/s41561-019-0510-8](https://doi.org/10.1038/s41561-019-0510-8)), ITS_LIVE v2,
   RACMO2.3p2 ANT27 (van Wessem et al., 2018, doi:[10.5194/tc-12-1479-2018](https://doi.org/10.5194/tc-12-1479-2018));
   the model domain of De Rydt et al. (2021, doi:[10.5194/tc-15-113-2021](https://doi.org/10.5194/tc-15-113-2021)),
   with the seabed near the grounding line deepened as in Bett et al. (2024,
   doi:[10.5194/tc-18-2653-2024](https://doi.org/10.5194/tc-18-2653-2024)); observed thinning rates of Smith et al.
   (2020, doi:[10.1126/science.aaz5845](https://doi.org/10.1126/science.aaz5845)). Mendenhall: the OGGM
   pre-processing chain (Maussion et al., 2019, doi:[10.5194/gmd-12-909-2019](https://doi.org/10.5194/gmd-12-909-2019))
   on the Copernicus GLO-30 DEM, with the consensus thickness of Farinotti et al. (2019,
   doi:[10.1038/s41561-019-0300-3](https://doi.org/10.1038/s41561-019-0300-3)) and the ITS_LIVE composite velocity
   (RGI60-01.00709); proglacial lake at 15 m.
2. **IGM input** on the model grid: bed, surface, thickness, surface mass balance, observed speed, masks, water
   level (ocean at 0 m, ρ_w = 1023 / 1027 kg m⁻³; lake at 15 m, 1000 kg m⁻³), uniform rate factor (Table C1).
3. **Basal friction.** The Weertman coefficient τ_ref was calibrated cell by cell against the observed surface
   speed, by a fixed-point iteration with one cold Newton–CG solve per update (110 updates); floating ice is
   frictionless.
4. **Sub-shelf melt** (ocean glaciers): the coefficient of the quadratic local law fitted to the observed mean melt
   (Sermeq Kujalleq: 230 m yr⁻¹ on the tongue, Motyka et al., 2011; Pine Island: ISMIP6 form, γ₀ = 49 500 m yr⁻¹).
5. **Flowline** from the trunk to the shelf, for the time series of `marine_ts`.
6. **Relaxation** under present-day forcing with a fixed front: 2 yr (Sermeq Kujalleq), 5 yr (Pine Island,
   Mendenhall). The runs start from the end of the relaxation, at t = 0. On Pine Island the relaxation was iterated
   with the thermal-forcing and flux corrections (after Hill et al., 2023) recomputed from the observed thinning
   rates, so that the initial state is close to balance.
7. **Ice thinner than 1 cm** is removed on Sermeq Kujalleq and Pine Island (`thin_ice`), since cells with vanishing
   ice have undetermined velocities that collapse the CFL time step.

The raw data and the preparation scripts are not part of this repository; the paper's Appendix C and
[`../how_the_results_were_obtained.md`](../how_the_results_were_obtained.md) (Sect. 2) describe every setting, and the
input files above are what the runs used.
