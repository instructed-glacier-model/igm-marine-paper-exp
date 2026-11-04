# Unconfined ice shelf figure (Fig. 2)

```bash
python make_figure.py      # seconds; needs numpy, pandas, matplotlib
```

Writes `fig_02_albrecht_verification.{pdf,png}` (the PDF is Fig. 2 of the paper). Orange: sub-grid front; blue:
level-set front; dashed: the analytic references.

`data/` holds what the figure needs, written by `../runs/make_figure_data.py` from the runs:

| file | content | panel |
|---|---|---|
| `profiles_2.5km.csv` | `scheme`, `x_km`, `thickness_m`, `velocity_m_yr`: the ice of the last frame of `adv_<scheme>_2.5km` (middle row), and the analytic profiles (`scheme` = `analytic`) | (a), (b) |
| `steady_front.csv` | `dx_km`, `scheme`, `x_front_km`: the final front of `adv_<scheme>_<res>`; `analytic_km` = 144.45 | (c) |
| `trajectories.csv` | `run`, `scheme`, `t_yr`, `x_front_km`, `reference_km`: the front of the 5-km transient runs and its analytic reference | (d)–(f) |
