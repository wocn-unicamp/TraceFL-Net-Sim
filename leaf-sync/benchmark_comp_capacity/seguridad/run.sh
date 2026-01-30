#!/bin/bash
set -euo pipefail

# =========================
# Configuración
# =========================
PYTHON="${PYTHON:-python}"
BENCH="${BENCH:-bench_gemm.py}"

# Para “1 núcleo”: fijamos a un CPU lógico (0 por defecto)
CPU_CORE="${CPU_CORE:-0}"

# Para “multi-core”: fija un rango de CPUs lógicos (ajusta si quieres)
CPU_CORES_ALL="${CPU_CORES_ALL:-0-15}"   # tu máquina tiene 16 CPUs lógicos

# Valores de threads a probar
THREAD_VALUES=(1 2 4 8 16)

echo "=============================================="
echo " Benchmark GEMM (NumPy/BLAS)"
echo "  - Single-core pinned: CPU ${CPU_CORE}"
echo "  - Multi-core pinned : CPU ${CPU_CORES_ALL}"
echo "=============================================="

# =========================
# Función: mostrar env actual
# =========================
print_env () {
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
  echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-}"
  echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-}"
  echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-}"
}

# =========================
# Función: ejecutar benchmark
#   $1 = pinning (pin|nopin)
#   $2 = cpu_set (ej: 0, 0-7)  [solo si pin]
# =========================
run_bench () {
  local pinning="$1"
  local cpu_set="${2:-}"

  echo ""
  echo "----------------------------------------------"
  echo "PINNING=${pinning}  CPU_SET=${cpu_set:-<none>}"
  print_env
  echo "----------------------------------------------"

  if [[ "${pinning}" == "pin" ]]; then
    taskset -c "${cpu_set}" "${PYTHON}" "${BENCH}"
  else
    "${PYTHON}" "${BENCH}"
  fi
}

# =========================
# Caso 1: Single-core real (pinned + 1 thread)
# =========================
echo ""
echo ">>> Caso 1: Single-core pinned + 1 thread"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
run_bench pin "${CPU_CORE}"

# =========================
# Caso 2: Multi-core pinned (varía threads, permite escalar)
# =========================
for t in "${THREAD_VALUES[@]}"; do
  echo ""
  echo ">>> Caso 2: Multi-core pinned (${CPU_CORES_ALL}) + ${t} threads"
  export OMP_NUM_THREADS="${t}"
  export MKL_NUM_THREADS="${t}"
  export OPENBLAS_NUM_THREADS="${t}"
  export NUMEXPR_NUM_THREADS="${t}"
  run_bench pin "${CPU_CORES_ALL}"
done

# =========================
# Caso 3: Default real (sin variables y SIN pinning)
# =========================
echo ""
echo ">>> Caso 3: Default real (unset) + NO pinning"
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS
unset NUMEXPR_NUM_THREADS
run_bench nopin

echo ""
echo "=============================================="
echo " Benchmark finalizado"
echo "=============================================="
