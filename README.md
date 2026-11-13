# igm-marine-paper-exp

Material to reproduce the experiments and figures of the paper

> Gregov, T., Rosier, S. H. R., Leger, T. P. M., Finley, B. D., Vieli, A., and Jouvet, G.: Modeling marine- and
> lake-terminated glaciers on GPUs with physics-informed machine learning (IGM v3.3.0), submitted to *Geoscientific
> Model Development*.

IGM: <https://github.com/instructed-glacier-model/igm> — documentation: <https://igm-model.org/>

## Contents

One folder per experiment, each with `runs/` (to re-run it: configurations, user modules, inputs, run script) and
`figure/` (to redraw the figure from the derived data shipped there: minutes on a laptop, no GPU).

| Experiment | Paper | Instructions |
|---|---|---|
| Schematic of a marine-terminated glacier | Fig. 1 | [`schematic/README.md`](schematic/README.md) |
| Calving-front motion on an unconfined ice shelf (Albrecht et al., 2011) | Sect. 4.1, Fig. 2 | [`unconfined_ice_shelf/runs/README.md`](unconfined_ice_shelf/runs/README.md) |
| Grounding-line migration, MISMIP+ (Asay-Davis et al., 2016) | Sect. 4.2, Fig. 3 | [`mismip_plus/runs/README.md`](mismip_plus/runs/README.md) |
| Three glaciers: Sermeq Kujalleq, Pine Island Glacier, Mendenhall Glacier | Sect. 5, Figs. 4–6, 8 and C1, Tables 1, 2, C1 | [`glaciers/runs/README.md`](glaciers/runs/README.md) |
| Cost of the reference solver: scaling, and comparison with PISM | Sect. 5.4, App. D, Fig. 7, Table D1 | [`solver_cost/runs/README.md`](solver_cost/runs/README.md) |

## Redraw the figures

```bash
pip install -r requirements.txt       # numpy, pandas, matplotlib, xarray, netCDF4, cmcrameri
python unconfined_ice_shelf/figure/make_figure.py
python mismip_plus/figure/make_figure.py
python glaciers/figure/make_figures.py
python solver_cost/figure/make_figure.py
```

Each script writes its figure next to itself, under the file name of the manuscript (`fig_02_…pdf`, …); the shipped
PDFs are these. Figs. 7 and C1 exist in two layouts: `*_wide.pdf` (panels side by side) for the single-column
manuscript, and the stacked `fig_07_solver_cost.pdf` / `fig_C1_flowline_profiles.pdf` for the two-column version. Fig. 1 is built with `pdflatex`. The figures are set in Nimbus Sans (package `fonts-urw-base35`);
without it the scripts fall back to Helvetica, Arial or DejaVu Sans.

## IGM version

IGM v3.3.0:

```bash
pip install igm-model==3.3.0
```

All IGM runs used one NVIDIA RTX 4090 (24 GiB) per run, Python 3.10 and TensorFlow 2.17; the PISM runs of the
comparison used 32 cores of a dual Intel Xeon E5-2683 v4 node (`solver_cost/runs/pism/README.md`).

## License

GNU General Public License v3 (see [`LICENSE`](LICENSE)), as IGM. Third-party data keep the terms of their sources,
cited in the README of each experiment.
