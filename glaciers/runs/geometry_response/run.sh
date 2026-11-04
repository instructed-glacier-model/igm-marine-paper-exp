#!/usr/bin/env bash
# The runs behind Fig. 8 (Sect. 5.5): the error split of the N = 20 neural-network runs. Run from glaciers/runs after
# run_all.sh (it needs outputs/REFERENCE and outputs/RUN of each glacier), in an environment with IGM, one GPU per run.
# Three kinds of runs per glacier, with REFERENCE the reference run and RUN the N = 20 network run:
#   w_RUN                  the network run again, saving its weights at every frame (bitwise the original run);
#   netreplay_exact_RUN    the network of each frame evaluated on the reference state of that frame, without training;
#   replay_REFERENCE, replay_RUN   cold Newton-CG solves of every saved frame (snapshot numerics: CG tolerance 1e-4,
#                          halt at 0.1 m/yr, single precision, probe_batch 9).
# The replays start from an input file holding the frame-0 geometry of the source run (prepare_reference_input.py).
# Then: python geometry_response/geometry_response.py  ->  ../figure/data/geometry_response.csv
set -euo pipefail
cd "$(dirname "$0")/.."   # glaciers/runs
PY=${PY:-python}
NET="processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0"
REPLAY="processes.iceflow.numerics.precision=single processes.iceflow.unified.cg_newton.probe_batch=9"

# ---- Sermeq Kujalleq (25 yr) ----
igm_run +experiment=glacier_jak_P1_30_T2_init_stop_w processes.time.end=25.0 processes.thk.front.min_thickness=5.0 $NET processes.iceflow.unified.nbit=[[0.0,25],[1.0,20]] hydra.run.dir=outputs/w_j25_n25_20_mt5
$PY geometry_response/prepare_reference_input.py outputs/glacier_jak_P1_30_exact_fast_mt5_25yr 0 outputs/glacier_jak_P1_30_exact_fast_mt5_25yr/reference_input_0.nc
igm_run +experiment=glacier_jak_netreplay inputs.local.filename=../outputs/glacier_jak_P1_30_exact_fast_mt5_25yr/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/glacier_jak_P1_30_exact_fast_mt5_25yr processes.load_network_weights.source=outputs/w_j25_n25_20_mt5 hydra.run.dir=outputs/netreplay_exact_j25_n25_20_mt5
igm_run +experiment=glacier_jak_replay inputs.local.filename=../outputs/glacier_jak_P1_30_exact_fast_mt5_25yr/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/glacier_jak_P1_30_exact_fast_mt5_25yr $REPLAY hydra.run.dir=outputs/replay_glacier_jak_P1_30_exact_fast_mt5_25yr
$PY geometry_response/prepare_reference_input.py outputs/j25_n25_20_mt5 0 outputs/j25_n25_20_mt5/reference_input_0.nc
igm_run +experiment=glacier_jak_replay inputs.local.filename=../outputs/j25_n25_20_mt5/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/j25_n25_20_mt5 $REPLAY hydra.run.dir=outputs/replay_j25_n25_20_mt5

# ---- Pine Island Glacier (100 yr) ----
igm_run +experiment=glacier_pig_W1_T2_init_stop_w processes.time.end=100.0 processes.thk.front.min_thickness=5.0 $NET processes.iceflow.unified.nbit=[[0.0,50],[1.0,20]] hydra.run.dir=outputs/w_pig_n50_20_mt5_100yr
$PY geometry_response/prepare_reference_input.py outputs/glacier_pig_exact_fast_mt5_100yr 0 outputs/glacier_pig_exact_fast_mt5_100yr/reference_input_0.nc
igm_run +experiment=glacier_pig_netreplay inputs.local.filename=../outputs/glacier_pig_exact_fast_mt5_100yr/reference_input_0.nc processes.time.start=0 processes.time.end=100 processes.replay_geometry.source=outputs/glacier_pig_exact_fast_mt5_100yr processes.load_network_weights.source=outputs/w_pig_n50_20_mt5_100yr hydra.run.dir=outputs/netreplay_exact_pig_n50_20_mt5_100yr
igm_run +experiment=glacier_pig_replay inputs.local.filename=../outputs/glacier_pig_exact_fast_mt5_100yr/reference_input_0.nc processes.time.start=0 processes.time.end=100 processes.replay_geometry.source=outputs/glacier_pig_exact_fast_mt5_100yr $REPLAY hydra.run.dir=outputs/replay_glacier_pig_exact_fast_mt5_100yr
$PY geometry_response/prepare_reference_input.py outputs/pig_n50_20_mt5_100yr 0 outputs/pig_n50_20_mt5_100yr/reference_input_0.nc
# In single precision this replay slows to minutes per frame from 57 yr on the nearly detached floating block of the
# network geometry (CG at its iteration cap); it was stopped there (processes.time.end=57) and the frames from 57 yr
# were solved in double precision in replay_..._dbl. geometry_response.py takes each frame from whichever holds it.
igm_run +experiment=glacier_pig_replay inputs.local.filename=../outputs/pig_n50_20_mt5_100yr/reference_input_0.nc processes.time.start=0 processes.time.end=57 processes.replay_geometry.source=outputs/pig_n50_20_mt5_100yr $REPLAY hydra.run.dir=outputs/replay_pig_n50_20_mt5_100yr
$PY geometry_response/prepare_reference_input.py outputs/pig_n50_20_mt5_100yr 57 outputs/pig_n50_20_mt5_100yr/reference_input_57.nc
igm_run +experiment=glacier_pig_replay inputs.local.filename=../outputs/pig_n50_20_mt5_100yr/reference_input_57.nc processes.time.start=57 processes.time.end=100 processes.replay_geometry.source=outputs/pig_n50_20_mt5_100yr hydra.run.dir=outputs/replay_pig_n50_20_mt5_100yr_dbl

# ---- Mendenhall Glacier (25 yr) ----
igm_run +experiment=glacier_men_E1_T2_init_stop_w processes.time.end=25.0 processes.thk.front.min_thickness=5.0 $NET processes.iceflow.unified.nbit=[[0.0,50],[1.0,20]] hydra.run.dir=outputs/w_m25_n50_20_mt5
$PY geometry_response/prepare_reference_input.py outputs/glacier_men_exact_fast_mt5_25yr 0 outputs/glacier_men_exact_fast_mt5_25yr/reference_input_0.nc
igm_run +experiment=glacier_men_netreplay inputs.local.filename=../outputs/glacier_men_exact_fast_mt5_25yr/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/glacier_men_exact_fast_mt5_25yr processes.load_network_weights.source=outputs/w_m25_n50_20_mt5 hydra.run.dir=outputs/netreplay_exact_m25_n50_20_mt5
igm_run +experiment=glacier_men_replay inputs.local.filename=../outputs/glacier_men_exact_fast_mt5_25yr/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/glacier_men_exact_fast_mt5_25yr $REPLAY hydra.run.dir=outputs/replay_glacier_men_exact_fast_mt5_25yr
$PY geometry_response/prepare_reference_input.py outputs/m25_n50_20_mt5 0 outputs/m25_n50_20_mt5/reference_input_0.nc
igm_run +experiment=glacier_men_replay inputs.local.filename=../outputs/m25_n50_20_mt5/reference_input_0.nc processes.time.start=0 processes.time.end=25 processes.replay_geometry.source=outputs/m25_n50_20_mt5 $REPLAY hydra.run.dir=outputs/replay_m25_n50_20_mt5

$PY geometry_response/geometry_response.py
