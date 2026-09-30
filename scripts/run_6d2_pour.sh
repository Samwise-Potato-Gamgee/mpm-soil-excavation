#!/usr/bin/env bash
# run_6d2_pour.sh -- Step 6d-test2 full-pour GPU driver (P0 then P1d).
#
# Sequential driver copied from run_6d2_feas.sh. GPU safety rules are unchanged: own log, own VRAM
# poll (every 2 s), RUN_DONE/RUN_FAILED sentinel, `timeout 600`, kill -9 at most 60 s after the
# sentinel, then a GPU-free check before the next run. Only PROJECT is absolute.
#
# A run ending in RUN_FAILED, a timeout, or a NaN is recorded in logs/phase3_6d2_pour_failures.txt
# and the driver CONTINUES with the next run, but only if no GPU process is left over; if a GPU
# process is left over (after kill -9 and re-check) the whole driver stops.
#
# Usage (no args = all runs):
#   bash scripts/run_6d2_pour.sh
#   bash scripts/run_6d2_pour.sh pour_P0

set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGS="$PROJECT/logs"
RUNNER="$PROJECT/scripts/mpm_pour_run.py"
CONDA_ENV="$PROJECT/envs/mpm"
FAILFILE="$LOGS/phase3_6d2_pour_failures.txt"

all_names=(
  pour_P0 pour_P1d
)

args_for() {
  case "$1" in
    pour_P0)  printf '%s\n' "--strain-basis P0" ;;
    pour_P1d) printf '%s\n' "--strain-basis P1d" ;;
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

  echo "[driver] start $name : mpm_pour_run.py $*"
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

  local sb status="OK" pour=""
  sb=$(printf '%s\n' "$*" | grep -o -- '--strain-basis [A-Za-z0-9]*' | awk '{print $2}')
  local steps stop nan m1 m2 m3 m4 active_end
  if [ "$sentinel" != "RUN_DONE" ]; then
    if [ "$rc" -eq 124 ]; then status="TIMEOUT"; else status="$sentinel"; fi
  else
    pour=$(grep '^\[POUR\]' "$log" | tail -n1)
    steps=$(printf '%s\n' "$pour" | grep -o 'steps=[0-9]*' | cut -d= -f2)
    stop=$(printf '%s\n' "$pour" | grep -o 'stop=[A-Za-z_]*' | cut -d= -f2)
    nan=$(printf '%s\n' "$pour" | grep -o 'nan=[A-Za-z]*' | cut -d= -f2)
    m1=$(printf '%s\n' "$pour" | grep -o 'med_q1=[0-9.]*' | cut -d= -f2)
    m2=$(printf '%s\n' "$pour" | grep -o 'med_q2=[0-9.]*' | cut -d= -f2)
    m3=$(printf '%s\n' "$pour" | grep -o 'med_q3=[0-9.]*' | cut -d= -f2)
    m4=$(printf '%s\n' "$pour" | grep -o 'med_q4=[0-9.]*' | cut -d= -f2)
    active_end=$(printf '%s\n' "$pour" | grep -o 'active_end=[0-9]*' | cut -d= -f2)
    if [ "$nan" = "True" ]; then status="NAN"; fi
  fi

  echo "[driver] $name status=$status sbasis=${sb:-?} steps=${steps:-?} stop=${stop:-?} wall=${wall}s med_q1=${m1:-?} med_q2=${m2:-?} med_q3=${m3:-?} med_q4=${m4:-?} active_end=${active_end:-?} nan=${nan:-?} vram_peak_minus_idle=${dv}MiB"

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
