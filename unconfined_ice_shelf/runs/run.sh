#!/usr/bin/env bash
# The 15 runs of Sect. 4.1 of the paper (Fig. 2), in sequence on one GPU. Run from this folder, which holds
# experiment/, user/ and data/, in an environment with IGM (see ../../README.md). About 25 min on one RTX 4090.
#   adv_<scheme>_<res>   advance from a two-cell shelf to the steady state (0-500 yr), 250-m minimum thickness
#   advance_, retreat_   prescribed front velocities +200 / -300 m/yr from the 5-km steady state (300 yr)
#   const_, vm_          constant calving rate 800 m/yr and von Mises calving (0.09 MPa) from the same state
#   adv_none_5km         control without a front scheme (text only)
# The rate runs start from data/restart/adv_<scheme>_5km.nc, written by make_restart.py from the adv_ runs
# (the shipped files are those of the paper's runs). Then: python analysis.py (numbers of the text) and
# python make_figure_data.py (the CSV files of ../figure/data).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
for run in adv_sub_grid_5km adv_level_set_5km adv_sub_grid_10km adv_level_set_10km adv_sub_grid_2.5km adv_level_set_2.5km adv_none_5km; do
  igm_run +experiment=$run hydra.run.dir=outputs/$run
done
$PY make_restart.py adv_sub_grid_5km
$PY make_restart.py adv_level_set_5km
for run in advance_sub_grid_5km retreat_sub_grid_5km const_sub_grid_5km vm_sub_grid_5km advance_level_set_5km retreat_level_set_5km const_level_set_5km vm_level_set_5km; do
  igm_run +experiment=$run hydra.run.dir=outputs/$run
done
$PY analysis.py
$PY make_figure_data.py
