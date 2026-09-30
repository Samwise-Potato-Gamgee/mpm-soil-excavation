#!/usr/bin/env bash
# run_6d1_study.sh -- Step 6d-test1 GPU driver (dilatancy and strain-basis sweeps, voxel 0.015 m).
#
# Copy of run_6c2_study.sh with the Step-6d-test1 run list. GPU safety rules are unchanged:
# own log, own VRAM poll (every 2 s), RUN_DONE/RUN_FAILED sentinel, `timeout 600`, kill -9 at most
# 60 s after the sentinel, then a GPU-free check before the next run. Only PROJECT is absolute.
#
# Difference from run_6c2_study.sh: a run ending in RUN_FAILED, a timeout, or a NaN is recorded in
# logs/phase3_6d1_failures.txt and the driver CONTINUES with the next run, but only if no GPU
# process is left over; if a GPU process is left over (after kill -9 and re-check) the whole driver
# stops. After each run it prints a one-line summary including the first/last-third medians.
#
# Usage (no args = all runs):
#   bash scripts/run_6d1_study.sh
#   bash scripts/run_6d1_study.sh d_regr_default d_regr_explicit
#   bash scripts/run_6d1_study.sh e_dil01_h10 e_Q1_h25 ...

set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGS="$PROJECT/logs"
RUNNER="$PROJECT/scripts/mpm_run.py"
ANALYSIS="$PROJECT/scripts/mpm_resolution_study.py"
CONDA_ENV="$PROJECT/envs/mpm"
FAILFILE="$LOGS/phase3_6d1_failures.txt"
COMMON="--scene A --diameter 0.15 --duration 3.0 --seed 0 --soil-phi-deg 34 --ground-friction 0.5 --ground-half 1.5 --voxel 0.015"

all_names=(
  d_regr_default d_regr_explicit
  e_dil01_h10 e_dil03_h10 e_dil10_h10 e_P1d_h10 e_Q1_h10
  e_dil01_h25 e_dil03_h25 e_dil10_h25 e_P1d_h25 e_Q1_h25
)

args_for() {
  case "$1" in
    d_regr_default)  printf '%s\n' "$COMMON --height-factor 1.0" ;;
    d_regr_explicit) printf '%s\n' "$COMMON --height-factor 1.0 --dilatancy 0.0 --strain-basis P0" ;;
    e_dil01_h10)     printf '%s\n' "$COMMON --height-factor 1.0 --dilatancy 0.1" ;;
    e_dil03_h10)     printf '%s\n' "$COMMON --height-factor 1.0 --dilatancy 0.3" ;;
    e_dil10_h10)     printf '%s\n' "$COMMON --height-factor 1.0 --dilatancy 1.0" ;;
    e_P1d_h10)       printf '%s\n' "$COMMON --height-factor 1.0 --strain-basis P1d" ;;
    e_Q1_h10)        printf '%s\n' "$COMMON --height-factor 1.0 --strain-basis Q1" ;;
    e_dil01_h25)     printf '%s\n' "$COMMON --height-factor 2.5 --dilatancy 0.1" ;;
    e_dil03_h25)     printf '%s\n' "$COMMON --height-factor 2.5 --dilatancy 0.3" ;;
    e_dil10_h25)     printf '%s\n' "$COMMON --height-factor 2.5 --dilatancy 1.0" ;;
    e_P1d_h25)       printf '%s\n' "$COMMON --height-factor 2.5 --strain-basis P1d" ;;
    e_Q1_h25)        printf '%s\n' "$COMMON --height-factor 2.5 --strain-basis Q1" ;;
    *) printf '%s\n' "" ;;
  esac
}

check_gpu_free() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null)
  if [ -n "$apps" ]; then
    echo "[driver] GPU still has compute apps:"
    printf '%s\n' "$apps"
    printf '%s\n' "$apps" | grep -i "python" | cut -d, -f1 | tr -d ' ' | while read -r p; do
      if [ -n "$p" ]; then kill -9 "$p" 2>/dev/null || true; fi
    done
    sleep 2
    apps=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null)
    if [ -n "$apps" ]; then
      echo "[driver] ABORT: GPU not free after run"
      return 1
    fi
  fi
  return 0
}

run_one() {
  local name="$1"; shift
  local log="$LOGS/${name}.log"
  local vram="$LOGS/${name}_vram.csv"
  local idle="$LOGS/${name}_idle.txt"

  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 > "$idle" || echo "0" > "$idle"
  : > "$vram"

  # dilatancy / strain basis as passed on the command line (defaults otherwise)
  local dil sb
  dil=$(printf '%s\n' "$*" | grep -o -- '--dilatancy [0-9.]*' | awk '{print $2}')
  [ -z "$dil" ] && dil=0.0
  sb=$(printf '%s\n' "$*" | grep -o -- '--strain-basis [A-Za-z0-9]*' | awk '{print $2}')
  [ -z "$sb" ] && sb=P0

  echo "[driver] start $name : mpm_run.py $*"
  local t0 t1 wall
  t0=$(date +%s.%N)

  timeout 600 conda run --no-capture-output --prefix "$CONDA_ENV" python "$RUNNER" "$@" --out-prefix "$name" > "$log" 2>&1 &
  local pid=$!

  local sentinel=""
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 >> "$vram"
    if grep -q "RUN_DONE" "$log" 2>/dev/null; then sentinel="RUN_DONE"; break; fi
    if grep -q "RUN_FAILED" "$log" 2>/dev/null; then sentinel="RUN_FAILED"; break; fi
    sleep 2
  done

  if [ -z "$sentinel" ]; then
    if grep -q "RUN_DONE" "$log" 2>/dev/null; then sentinel="RUN_DONE"
    elif grep -q "RUN_FAILED" "$log" 2>/dev/null; then sentinel="RUN_FAILED"
    else sentinel="NO_SENTINEL"; fi
  fi

  local waited=0
  if kill -0 "$pid" 2>/dev/null; then
    while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 2; waited=$((waited+2)); done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[driver] $name still alive ${waited}s after sentinel -> kill -9"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  local rc=0
  wait "$pid" 2>/dev/null || rc=$?
  t1=$(date +%s.%N)
  wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')

  local peak=0 idle_v=0 dv
  if [ -s "$vram" ]; then peak=$(sort -n "$vram" | tail -n1); fi
  if [ -s "$idle" ]; then idle_v=$(head -n1 "$idle"); fi
  dv=$(awk -v p="$peak" -v i="$idle_v" 'BEGIN{printf "%.0f", p-i}')

  local nsteps nparts med
  nsteps=$(grep -o "n_steps=[0-9]*" "$log" | head -n1 | cut -d= -f2)
  nparts=$(grep -o "N=[0-9]*" "$log" | tail -n1 | cut -d= -f2)
  med=$(grep -o "median_step_ms=[0-9.]*" "$log" | tail -n1 | cut -d= -f2)

  # status + first/last-third medians + NaN (from the CPU analysis one-line mode)
  local status="OK" first="" last="" nan=""
  if [ "$sentinel" != "RUN_DONE" ]; then
    if [ "$rc" -eq 124 ]; then status="TIMEOUT"; else status="$sentinel"; fi
  else
    local oneline
    oneline=$(conda run --no-capture-output --prefix "$CONDA_ENV" python "$ANALYSIS" --one-line "$name" 2>/dev/null | grep '^ONE_LINE' | tail -n1)
    nan=$(printf '%s\n' "$oneline" | grep -o 'nan=[A-Za-z]*' | cut -d= -f2)
    first=$(printf '%s\n' "$oneline" | grep -o 'first_ms=[0-9.]*' | cut -d= -f2)
    last=$(printf '%s\n' "$oneline" | grep -o 'last_ms=[0-9.]*' | cut -d= -f2)
    if [ "$nan" = "True" ]; then status="NAN"; fi
  fi

  echo "[driver] $name status=$status dil=$dil sbasis=$sb N=${nparts:-?} steps=${nsteps:-?} wall=${wall}s median_step_ms=${med:-?} first3_ms=${first:-?} last3_ms=${last:-?} vram_peak_minus_idle=${dv}MiB"

  if [ "$status" != "OK" ]; then
    printf '%s %s\n' "$name" "$status" >> "$FAILFILE"
  fi
  return 0
}

: > "$FAILFILE"

names=("$@")
if [ "${#names[@]}" -eq 0 ]; then
  names=("${all_names[@]}")
fi

for name in "${names[@]}"; do
  args=$(args_for "$name")
  if [ -z "$args" ]; then
    echo "[driver] ABORT: unknown run name '$name'"
    exit 1
  fi
  run_one "$name" $args
  if ! check_gpu_free; then
    echo "[driver] GPU not free after $name -> stopping the whole driver"
    printf '%s %s\n' "$name" "GPU_LEFTOVER" >> "$FAILFILE"
    exit 1
  fi
done

echo "[driver] all requested runs finished"
