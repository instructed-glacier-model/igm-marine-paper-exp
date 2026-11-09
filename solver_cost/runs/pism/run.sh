#!/usr/bin/env bash
# The eight PISM cases of Fig. 7 and Table D1: SSA, and Blatter-Pattyn with two vertical levels, on the four Pine
# Island grids (../data/<res>/initial.nc), 32 MPI ranks bound one per physical core, run one after the other.
# Before: ./setup.sh (environments), ./build_bp_front_extension.sh (the adapter of the BP runs), source pism_env.sh.
# Each case writes results/<case>/: summary.json (every solve: time, iterations, memory, convergence), audit.json
# (the geometry mapping), geometry.nc and the velocity fields initialization_0.nc and cold_1..5.nc.
# The recorded cases took 4 s (SSA 4 km) to 10 min (BP 0.5 km) of process time on a dual Xeon E5-2683 v4 node.
set -euo pipefail
cd "$(dirname "$0")"
source pism_env.sh
MPI="mpiexec --bind-to core --map-by core --report-bindings -n 32 python"
TOL="--rtol 1e-6 --ksp-rtol 1e-4"

# SSA (Q1 finite elements; the quasi-rigid band as a stationary boundary; IGM's lower surface and grounding mask)
for res in 4km 2km 1km 500m; do
  extra=""
  if [ "$res" = 500m ]; then extra="-pc_gamg_reuse_interpolation false"; fi
  $MPI pism_diag.py --out results/ssa_${res}_p32 --model ssa --reps 5 $TOL --initial ../data/$res/initial.nc \
    --band-bc stationary --geometry-mode ice-base $extra
done

# Blatter-Pattyn, two vertical levels (the band keeps its drag; IGM's front classification through the adapter)
for res in 4km 2km 1km 500m; do
  $MPI bp_diag.py --out results/bp_${res}_p32_z2 --model bp --nz 2 --reps 5 --band-bc native --geometry-mode ice-base \
    --front-treatment igm-wet --bp-preconditioner gamg $TOL --initial ../data/$res/initial.nc \
    -bp_pc_gamg_reuse_interpolation false
done
