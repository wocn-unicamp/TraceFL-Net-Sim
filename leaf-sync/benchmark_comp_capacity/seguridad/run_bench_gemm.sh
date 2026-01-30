#!/bin/bash
set -euo pipefail

# =========================
# Configuración
# =========================
PYTHON="${PYTHON:-python3}"
BENCH="${BENCH:-bench_gemm.py}"

# Para “1 núcleo”: fijamos a un CPU lógico (0 por defecto)
CPU_CORE="${CPU_CORE:-0}"

# Detecta CPUs lógicos disponibles (respeta cgroups/quotas)
NCPU="$(nproc)"

# Para “multi-core”: fija un rango de CPUs lógicos.
# Si el usuario define CPU_CORES_ALL, se respeta; si no, usa todos.
CPU_CORES_ALL="${CPU_CORES_ALL:-0-$((NCPU-1))}"

# -------------------------
# Generar lista de threads automáticamente: 1,2,4,8,... <= NCPU (+NCPU si aplica)
# -------------------------
THREAD_VALUES=()
t=1
while (( t <= NCPU )); do
  THREAD_VALUES+=("$t")
  t=$(( t * 2 ))
done
if [[ "${THREAD_VALUES[-1]}" -ne "$NCPU" ]]; then
  THREAD_VALUES+=("$NCPU")
fi

echo "=============================================="
echo " Benchmark GEMM (NumPy/BLAS)"
echo "  - Python            : ${PYTHON}"
echo "  - Single-core pinned: CPU ${CPU_CORE}"
echo "  - Multi-core pinned : CPU ${CPU_CORES_ALL} (nproc=${NCPU})"
echo "  - Thread values     : ${THREAD_VALUES[*]}"
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
for thr in "${THREAD_VALUES[@]}"; do
  echo ""
  echo ">>> Caso 2: Multi-core pinned (${CPU_CORES_ALL}) + ${thr} threads"
  export OMP_NUM_THREADS="${thr}"
  export MKL_NUM_THREADS="${thr}"
  export OPENBLAS_NUM_THREADS="${thr}"
  export NUMEXPR_NUM_THREADS="${thr}"
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
