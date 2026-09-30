#!/usr/bin/env bash
# run_6d2_feas.sh -- Step 6d-test2 pour-feasibility GPU driver (3 short runs, voxel 0.015 m).
#
# Sequential driver copied from run_6d1_study.sh. GPU safety rules are unchanged: own log, own VRAM
# poll (every 2 s), RUN_DONE/RUN_FAILED sentinel, `timeout 600`, kill -9 at most 60 s after the
# sentinel, then a GPU-free check before the next run. Only PROJECT is absolute.
#
# A run ending in RUN_FAILED, a timeout, or a NaN is recorded in logs/phase3_6d2_failures.txt and
# the driver CONTINUES with the next run, but only if no GPU process is left over; if a GPU process
# is left over (after kill -9 and re-check) the whole driver stops. After each run it prints a
# one-line summary.
#
# Usage (no args = all runs):
#   bash scripts/run_6d2_feas.sh
#   bash scripts/run_6d2_feas.sh feas_P1d_tall

set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGS="$PROJECT/logs"
RUNNER="$PROJECT/scripts/mpm_pour_feasibility.py"
CONDA_ENV="$PROJECT/envs/mpm"
FAILFILE="$LOGS/phase3_6d2_failures.txt"

all_names=(
  feas_P1d_tall feas_P1d_compact feas_P0_compact
)

args_for() {
  case "$1" in
    feas_P1d_tall)    printf '%s\n' "--strain-basis P1d --parking tall --steps 300" ;;
    feas_P1d_compact) printf '%s\n' "--strain-basis P1d --parking compact --steps 300" ;;
    feas_P0_compact)  printf '%s\n' "--strain-basis P0 --parking compact --steps 300" ;;
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

  echo "[driver] start $name : mpm_pour_feasibility.py $*"
  local t0 t1 wall
  t0=$(date +%s.%N)

  timeout 600 conda run --no-capture-output --prefix "$CONDA_ENV" python "$RUNNER" "$@" --prefix "$name" > "$log" 2>&1 &
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

  local sb parking steps nan med0 med50 medemit active_end n_below
  sb=$(printf '%s\n' "$*" | grep -o -- '--strain-basis [A-Za-z0-9]*' | awk '{print $2}')
  parking=$(printf '%s\n' "$*" | grep -o -- '--parking [A-Za-z]*' | awk '{print $2}')
  steps=$(printf '%s\n' "$*" | grep -o -- '--steps [0-9]*' | awk '{print $2}')

  local status="OK" feas=""
  if [ "$sentinel" != "RUN_DONE" ]; then
    if [ "$rc" -eq 124 ]; then status="TIMEOUT"; else status="$sentinel"; fi
  else
    feas=$(grep '^\[FEAS\]' "$log" | tail -n1)
    nan=$(printf '%s\n' "$feas" | grep -o 'nan=[A-Za-z]*' | cut -d= -f2)
    med0=$(printf '%s\n' "$feas" | grep -o 'med_step_0_49_ms=[0-9.]*' | cut -d= -f2)
    med50=$(printf '%s\n' "$feas" | grep -o 'med_step_50_299_ms=[0-9.]*' | cut -d= -f2)
    medemit=$(printf '%s\n' "$feas" | grep -o 'med_emit_ms=[0-9.]*' | cut -d= -f2)
    active_end=$(printf '%s\n' "$feas" | grep -o 'active_end=[0-9]*' | cut -d= -f2)
    n_below=$(printf '%s\n' "$feas" | grep -o 'n_below=[0-9]*' | cut -d= -f2)
    if [ "$nan" = "True" ]; then status="NAN"; fi
  fi

  echo "[driver] $name status=$status sbasis=${sb:-?} parking=${parking:-?} steps=${steps:-?} wall=${wall}s med_step_0_49_ms=${med0:-?} med_step_50_299_ms=${med50:-?} med_emit_ms=${medemit:-?} active_end=${active_end:-?} n_below=${n_below:-?} vram_peak_minus_idle=${dv}MiB"

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
