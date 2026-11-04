# MISMIP+ figure (Fig. 3)

```bash
python make_figure.py      # seconds; needs numpy, pandas, matplotlib
```

Writes `fig_03_mismip_timeseries.{pdf,png}` (the PDF is Fig. 3 of the paper). Orange: melt maintained (Ice{k}r then
Ice{k}rr); blue: melt stopped at 100 yr (Ice{k}ra). Shading: range of the 24 models of the main subset of Cornford
et al. (2020) (light) and of its 11 Weertman models (dark).

`data/` holds what the figure needs, written by `../runs/make_figure_data.py`:

| file | content | panel |
|---|---|---|
| `igm_timeseries.csv` | `exp`, `t_yr`, `xGL_centre_m`, `groundedArea_m2`, `iceVAF_m3` of the six IGM runs with the neural-network mapping (`outputs/<exp>_1km_nn/mismip_ts.csv`) | lines |
| `ensemble_envelope.csv` | for `k` (Ice1, Ice2), `branch` (`on`: r then rr; `off`: ra), `quantity` (`gl`, `area`, `vaf`) and `subset` (`main`, `weertman`): `min` and `max` over the `n` members every 10 yr; area and VAF as changes since t = 0 | shading |
| `igm_newton_cg.csv` | the same time series of the runs with the identity mapping solved with Newton–CG (the "less than 2 km" comparison of the text) | not drawn |
