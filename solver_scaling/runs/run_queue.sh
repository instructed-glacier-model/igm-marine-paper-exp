#!/usr/bin/env bash
# run_queue.sh GPU QUEUE_FILE [T_MAX_C] [MAX_WAIT_S]
# Runs the cases of a queue file on one GPU, from this folder: queue_main.txt (the 16 timed cases of Fig. 7),
# queue_ref.txt (the float64 references, needed by analyze.py), queue_probe.txt (the probe_batch comparison).
# Queue line: CASE_NAME MODE BASIS PRECISION RES [extra driver arguments...]
# Each case writes cases/CASE_NAME/ (summary_raw.json, repetitions_raw.csv, initial_solution.nc or reference.npz),
# the raw velocity fields of every solve to work/raw/CASE_NAME/ (needed by analyze.py), and a log to work/logs/.
# Thermal control, as for the paper: before each case the script waits until the GPU is at or below T_MAX_C
# (default 45 C; an idle RTX 4090 is at 35-37 C), for at most MAX_WAIT_S (default 900 s), and nvidia-smi logs the
# temperature, clock, power, utilisation and throttle reasons every second to work/logs/CASE_NAME.gpu.csv.
# Environment: PY (python with IGM; default python); IGM_SRC (optional IGM source tree put first on PYTHONPATH).
set -uo pipefail
GPU=$1; QUEUE=$2; TMAX=${3:-45}; MAXWAIT=${4:-900}
S=$(cd "$(dirname "$0")" && pwd)
cd "$S"
PY=${PY:-python}
if [ -n "${IGM_SRC:-}" ]; then export PYTHONPATH="$IGM_SRC${PYTHONPATH:+:$PYTHONPATH}"; fi
mkdir -p work/logs work/raw cases
temp() { nvidia-smi --id="$GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d ' '; }
while read -r NAME MODE BASIS PREC RES EXTRA; do
  [[ -z "${NAME:-}" || "$NAME" == \#* ]] && continue
  LOG=work/logs/$NAME.log
  if [ -f "cases/$NAME/summary_raw.json" ]; then echo "skip $NAME (done)"; continue; fi
  waited=0
  while [ "$(temp)" -gt "$TMAX" ] && [ "$waited" -lt "$MAXWAIT" ]; do sleep 10; waited=$((waited + 10)); done
  {
    echo "case $NAME on $(hostname) gpu $GPU, start $(date -Is)"
    echo "thermal: start temperature $(temp) C after waiting ${waited} s (threshold ${TMAX} C, max wait ${MAXWAIT} s)"
    echo "gpu apps before: $(nvidia-smi --id=$GPU --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  } > "$LOG"
  nvidia-smi --id="$GPU" --query-gpu=timestamp,temperature.gpu,clocks.sm,power.draw,utilization.gpu,clocks_throttle_reasons.active \
    --format=csv,noheader -l 1 > "work/logs/$NAME.gpu.csv" 2>/dev/null &
  SMI=$!
  CUDA_VISIBLE_DEVICES=$GPU TF_CPP_MIN_LOG_LEVEL=3 PYTHONUNBUFFERED=1 \
    $PY pig_diag_bench.py --mode "$MODE" --basis "$BASIS" --precision "$PREC" \
    --initial "data/$RES/initial.nc" --config run_config.yaml \
    --out "cases/$NAME" --raw "work/raw/$NAME" --gpu-index "$GPU" $EXTRA >> "$LOG" 2>&1
  rc=$?
  kill "$SMI" 2>/dev/null
  echo "rc $rc, end $(date -Is), end temperature $(temp) C" >> "$LOG"
done < "$QUEUE"
echo "queue $QUEUE finished $(date -Is)" >> "work/logs/$(basename "$QUEUE" .txt).done"
