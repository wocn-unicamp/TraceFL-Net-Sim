#!/bin/bash
set -euo pipefail

# ==========================================================
# GEMM benchmark driver (NumPy/BLAS)
# - No CLI parameters required (run: bash run_bench_gemm_v2.sh)
# - Produces one CSV per execution (run_id-based)
# ==========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON="${PYTHON:-python3}"
BENCH="${BENCH:-${SCRIPT_DIR}/bench_gemm_v3.py}"

# Pinning config
CPU_CORE="${CPU_CORE:-0}"

# Detect available logical CPUs (respects cgroups/quotas)
NCPU_TOTAL="$(nproc)"

# Multi-core CPU set (defaults to all available CPUs)
CPU_CORES_ALL="${CPU_CORES_ALL:-0-$((NCPU_TOTAL-1))}"

# If taskset exists, compute how many CPUs are in CPU_CORES_ALL (important if user overrides it)
if command -v taskset >/dev/null 2>&1; then
  NCPU_MULTI="$(taskset -c "${CPU_CORES_ALL}" nproc)"
else
  NCPU_MULTI="${NCPU_TOTAL}"
fi

# Benchmark parameters (override via environment if desired)
export BENCH_WARMUP="${BENCH_WARMUP:-2}"
export BENCH_ITERS="${BENCH_ITERS:-5}"
export BENCH_N_F32="${BENCH_N_F32:-4096}"
export BENCH_N_F64="${BENCH_N_F64:-3072}"
export BENCH_SEED="${BENCH_SEED:-12345}"

# Create a per-run CSV in the script directory
BENCH_RUN_ID="${BENCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export BENCH_RUN_ID

CSV_OUT="${CSV_OUT:-${SCRIPT_DIR}/bench_gemm_${BENCH_RUN_ID}.csv}"
export CSV_FILE="${CSV_OUT}"

# Thread values: 1,2,4,8,... <= NCPU_MULTI (+NCPU_MULTI if not already)
THREAD_VALUES=()
t=1
while (( t <= NCPU_MULTI )); do
  THREAD_VALUES+=("$t")
  t=$(( t * 2 ))
done
if [[ "${THREAD_VALUES[-1]}" -ne "${NCPU_MULTI}" ]]; then
  THREAD_VALUES+=("${NCPU_MULTI}")
fi

echo "=============================================="
echo " Benchmark GEMM (NumPy/BLAS)"
echo "  - Python            : ${PYTHON}"
echo "  - Bench script      : ${BENCH}"
echo "  - CSV output        : ${CSV_OUT}"
echo "  - Single-core pinned: CPU ${CPU_CORE}"
echo "  - Multi-core pinned : CPU ${CPU_CORES_ALL} (cpus=${NCPU_MULTI}, total_nproc=${NCPU_TOTAL})"
echo "  - Thread values     : ${THREAD_VALUES[*]}"
echo "  - Params            : warmup=${BENCH_WARMUP} iters=${BENCH_ITERS} n_f32=${BENCH_N_F32} n_f64=${BENCH_N_F64}"
echo "=============================================="

print_env () {
  echo "BENCH_THREADS=${BENCH_THREADS:-}"
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-}"
  echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-}"
  echo "OMP_DYNAMIC=${OMP_DYNAMIC:-}"
  echo "MKL_DYNAMIC=${MKL_DYNAMIC:-}"
}

run_bench () {
  local scenario="$1"     # label for CSV
  local pinning="$2"      # pin | nopin
  local cpu_set="${3:-}"  # only if pin

  export BENCH_SCENARIO="${scenario}"
  export BENCH_PINNING="${pinning}"
  export BENCH_CPUSET="${cpu_set}"

  echo ""
  echo "----------------------------------------------"
  echo "SCENARIO=${scenario}  PINNING=${pinning}  CPU_SET=${cpu_set:-<none>}"
  print_env
  echo "----------------------------------------------"

  if [[ "${pinning}" == "pin" ]]; then
    if command -v taskset >/dev/null 2>&1; then
      taskset -c "${cpu_set}" "${PYTHON}" "${BENCH}"
    else
      echo "WARN: taskset not found; running without pinning."
      "${PYTHON}" "${BENCH}"
    fi
  else
    "${PYTHON}" "${BENCH}"
  fi
}

# ----------------------------------------------------------
# Case 1: Single-core pinned + 1 thread
# ----------------------------------------------------------
echo ""
echo ">>> Case 1: Single-core pinned + 1 thread"
export BENCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export MKL_DYNAMIC=FALSE
run_bench "single_core_pinned" "pin" "${CPU_CORE}"

# ----------------------------------------------------------
# Case 2: Multi-core pinned + varying threads
# ----------------------------------------------------------
for thr in "${THREAD_VALUES[@]}"; do
  echo ""
  echo ">>> Case 2: Multi-core pinned (${CPU_CORES_ALL}) + ${thr} threads"
  export BENCH_THREADS="${thr}"
  export OMP_NUM_THREADS="${thr}"
  export MKL_NUM_THREADS="${thr}"
  export OPENBLAS_NUM_THREADS="${thr}"
  export NUMEXPR_NUM_THREADS="${thr}"
  export OMP_DYNAMIC=FALSE
  export MKL_DYNAMIC=FALSE
  run_bench "multi_core_pinned" "pin" "${CPU_CORES_ALL}"
done

# ----------------------------------------------------------
# Case 3: Default (unset) + no pinning
# ----------------------------------------------------------
echo ""
echo ">>> Case 3: Default (unset) + NO pinning"
unset BENCH_THREADS
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS
unset NUMEXPR_NUM_THREADS
unset OMP_DYNAMIC
unset MKL_DYNAMIC
run_bench "default_nopin" "nopin"

echo ""
echo "=============================================="
echo " Benchmark finished"
echo " CSV saved to: ${CSV_OUT}"
echo "=============================================="



echo "=============================================="
echo " Post-check: system back to default"
echo "=============================================="
echo "Shell CPU allowed list:"
grep Cpus_allowed_list /proc/$$/status || true

echo ""
echo "BLAS effective threads (best-effort):"
python3 - <<'PY'
import os
import numpy as np
print("NumPy:", np.__version__)
try:
    from threadpoolctl import threadpool_info
    blas = [d for d in threadpool_info() if d.get("user_api") == "blas"]
    for d in blas:
        print(f" - {d.get('internal_api')} : num_threads={d.get('num_threads')}  version={d.get('version')}")
except Exception as e:
    print("threadpoolctl not available:", e)
PY
