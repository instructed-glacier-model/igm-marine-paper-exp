#!/usr/bin/env bash
# The 15 runs of Sect. 5 of the paper: 3 glaciers x (1 reference run + 4 neural-network runs with N = 5, 10, 20, 50).
# Run from this folder (glaciers/runs), which holds experiment/, user/ and data/, in an environment with IGM
# (see ../../README.md), one GPU per run. Each run writes outputs/NAME/ (output.nc, marine_ts.csv, .hydra/config.yaml).
# The resolved configuration recorded for each run is configs/NAME.yaml, the authoritative record of every setting;
# configs/NAME_provenance.txt gives its command, host, GPU and the hashes of the IGM code, the input and the user code.
set -euo pipefail
cd "$(dirname "$0")"

# glacier_jak_P1_30_exact_fast_mt5_25yr: Sermeq Kujalleq, reference run (identity mapping, Newton-CG)
# input: glacier_jak/start_150m.nc
igm_run +experiment=glacier_jak_P1_30_exact_fast processes.time.end=25.0 processes.thk.front.min_thickness=5.0 hydra.run.dir=outputs/glacier_jak_P1_30_exact_fast_mt5_25yr

# j25_n25_5_mt5: Sermeq Kujalleq, neural-network run
# input: glacier_jak/start_150m.nc
igm_run +experiment=glacier_jak_P1_30_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,25],[1.0,5]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/j25_n25_5_mt5

# j25_n25_10_mt5: Sermeq Kujalleq, neural-network run
# input: glacier_jak/start_150m.nc
igm_run +experiment=glacier_jak_P1_30_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,25],[1.0,10]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/j25_n25_10_mt5

# j25_n25_20_mt5: Sermeq Kujalleq, neural-network run
# input: glacier_jak/start_150m.nc
igm_run +experiment=glacier_jak_P1_30_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,25],[1.0,20]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/j25_n25_20_mt5

# j25_n25_50_mt5: Sermeq Kujalleq, neural-network run
# input: glacier_jak/start_150m.nc
igm_run +experiment=glacier_jak_P1_30_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,25],[1.0,50]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/j25_n25_50_mt5

# glacier_pig_exact_fast_mt5_100yr: Pine Island Glacier, reference run (identity mapping, Newton-CG)
# input: glacier_pig/start_1km.nc
igm_run +experiment=glacier_pig_exact_fast_frames processes.time.end=100.0 processes.thk.front.min_thickness=5.0 processes.time.save=1.0 hydra.run.dir=outputs/glacier_pig_exact_fast_mt5_100yr

# pig_n50_5_mt5_100yr: Pine Island Glacier, neural-network run
# input: glacier_pig/start_1km.nc
igm_run +experiment=glacier_pig_W1_T2_init_stop processes.time.end=100.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,5]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/pig_n50_5_mt5_100yr

# pig_n50_10_mt5_100yr: Pine Island Glacier, neural-network run
# input: glacier_pig/start_1km.nc
igm_run +experiment=glacier_pig_W1_T2_init_stop processes.time.end=100.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,10]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/pig_n50_10_mt5_100yr

# pig_n50_20_mt5_100yr: Pine Island Glacier, neural-network run
# input: glacier_pig/start_1km.nc
igm_run +experiment=glacier_pig_W1_T2_init_stop processes.time.end=100.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,20]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/pig_n50_20_mt5_100yr

# pig_n50_50_mt5_100yr: Pine Island Glacier, neural-network run
# input: glacier_pig/start_1km.nc
igm_run +experiment=glacier_pig_W1_T2_init_stop processes.time.end=100.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,50]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/pig_n50_50_mt5_100yr

# glacier_men_exact_fast_mt5_25yr: Mendenhall Glacier, reference run (identity mapping, Newton-CG)
# input: glacier_men/start_50m.nc
igm_run +experiment=glacier_men_exact_fast_frames processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.time.save=1.0 hydra.run.dir=outputs/glacier_men_exact_fast_mt5_25yr

# m25_n50_5_mt5: Mendenhall Glacier, neural-network run
# input: glacier_men/start_50m.nc
igm_run +experiment=glacier_men_E1_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,5]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/m25_n50_5_mt5

# m25_n50_10_mt5: Mendenhall Glacier, neural-network run
# input: glacier_men/start_50m.nc
igm_run +experiment=glacier_men_E1_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,10]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/m25_n50_10_mt5

# m25_n50_20_mt5: Mendenhall Glacier, neural-network run
# input: glacier_men/start_50m.nc
igm_run +experiment=glacier_men_E1_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,20]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/m25_n50_20_mt5

# m25_n50_50_mt5: Mendenhall Glacier, neural-network run
# input: glacier_men/start_50m.nc
igm_run +experiment=glacier_men_E1_T2_init_stop processes.time.end=25.0 processes.thk.front.min_thickness=5.0 processes.iceflow.unified.nbit_init=4000 processes.iceflow.unified.nbit_warmup=-1 processes.iceflow.unified.nbit=[[0.0,50],[1.0,50]] processes.iceflow.unified.retrain_freq=2 processes.iceflow.unified.network.seed=0 hydra.run.dir=outputs/m25_n50_50_mt5
