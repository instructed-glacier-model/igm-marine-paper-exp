#!/usr/bin/env bash
# The MISMIP+ runs of Sect. 4.2 of the paper (Fig. 3), in sequence on one GPU. Run from this folder, which holds
# experiment/, user/ and data/, in an environment with IGM (see ../../README.md).
# Block 1, the neural-network mapping (the runs of Fig. 3): the Ice0 spin-up from a 100-m slab (10.8 kyr, ~7 min),
# whose end state is the Ice0 state data/restart/ice0_1km.nc (shipped; the restart files of the paper's runs are
# shipped too), then the Ice0 control and Ice1r / Ice2r (100 yr), and from their 100-yr state Ice1rr / Ice1ra and
# Ice2rr / Ice2ra. About 35 min of GPU time in all (1 km, RTX 4090).
# Block 2, the identity mapping solved with Newton-CG (the "less than 2 km" comparison of the text): the same seven
# experiments from the same Ice0 state, float64, about 2.5 h.
# Then: python make_figure_data.py (needs the ensemble of Cornford et al. 2020: python prep/fetch_ensemble.py).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
run() { igm_run +experiment="$1" hydra.run.dir="outputs/$1"; }

# --- block 1: neural-network mapping ---
run ice0_nn_1km
$PY make_restart.py ice0_nn_1km --out ice0_1km
run ice0_1km_nn
for k in 1 2; do
  run ice${k}r_1km_nn
  $PY make_restart.py ice${k}r_1km_nn --time 100 --out ice${k}r_1km_nn_t100
  run ice${k}rr_1km_nn
  run ice${k}ra_1km_nn
done

# --- block 2: identity mapping, Newton-CG ---
run ice0_1km
for k in 1 2; do
  run ice${k}r_1km
  $PY make_restart.py ice${k}r_1km --time 100 --out ice${k}r_1km_t100
  run ice${k}rr_1km
  run ice${k}ra_1km
done

$PY make_figure_data.py
