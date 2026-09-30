#!/usr/bin/env bash
# run_6c2_study.sh -- Step 6c-2 refinement-study GPU driver (voxel 0.01 m runs).
#
# Copy of run_6c_study.sh; only the run list changed. Runs the three 0.01 m jobs strictly one
# after another on the single shared RTX 4060 Ti. All GPU safety rules of run_6c_study.sh are kept:
# own log, own VRAM poll (every 2 s), RUN_DONE/RUN_FAILED sentinel, `timeout 600`, kill -9 at most
# 60 s after the sentinel, then a GPU-free check before the next run; aborts at the first failed
# run or leftover GPU process and prints the wall time of every run. Only PROJECT is absolute.
#
# Usage (no args = all runs):
#   bash scripts/run_6c2_study.sh
#   bash scripts/run_6c2_study.sh c_v010_h25

set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOGS="$PROJECT/logs"
RUNNER="$PROJECT/scripts/mpm_run.py"
CONDA_ENV="$PROJECT/envs/mpm"
COMMON="--diameter 0.15 --duration 3.0 --seed 0"

all_names=(
  c_v010_h05 c_v010_h10 c_v010_h25
)

args_for() {
  case "$1" in
    c_v010_h05)     printf '%s\n' "$COMMON --soil-phi-deg 34 --ground-friction 0.5 --ground-half 1.5 --voxel 0.01 --height-factor 0.5" ;;
    c_v010_h10)     printf '%s\n' "$COMMON --soil-phi-deg 34 --ground-friction 0.5 --ground-half 1.5 --voxel 0.01 --height-factor 1.0" ;;
    c_v010_h25)     printf '%s\n' "$COMMON --soil-phi-deg 34 --ground-friction 0.5 --ground-half 1.5 --voxel 0.01 --height-factor 2.5" ;;
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
  wait "$pid" 2>/dev/null
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

  echo "[driver] $name $sentinel N=${nparts:-?} steps=${nsteps:-?} wall=${wall}s median_step_ms=${med:-?} vram_peak_minus_idle=${dv}MiB"

  if [ "$sentinel" != "RUN_DONE" ]; then
    echo "[driver] ABORT: $name did not finish (sentinel=$sentinel)"
    return 1
  fi
  return 0
}

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
  if ! run_one "$name" $args; then
    echo "[driver] stopping after failed run $name"
    exit 1
  fi
  if ! check_gpu_free; then
    echo "[driver] stopping after $name (GPU not free)"
    exit 1
  fi
done

echo "[driver] all requested runs finished"
