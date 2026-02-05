# ============================================================
# (1) BENCHMARK “MODO ESTRITO”: 1 core fixo (CPU 0) + 1 thread
#
# O que este teste mede:
# - Desempenho de GEMM (multiplicação de matrizes) via NumPy/BLAS.
#
# Como está configurado:
# - PINNING: força execução somente no CPU lógico 0 (taskset -c 0).
# - THREADS: 1 thread para OMP/MKL/OpenBLAS/NumExpr.
# - DINÂMICA: OMP/MKL/OpenBLAS sem auto-ajuste de threads (dynamic OFF).
# - KMP_*: reduz overhead de runtime (principalmente quando BLAS = MKL).
# - TF_*: definido por consistência com o ambiente do LEAF (não afeta NumPy).
#
# Quando usar:
# - Para reproduzir ao máximo um cenário “single-core” com afinidade fixa,
#   típico de execuções controladas/replicáveis (como no seu comando do LEAF).
# ============================================================

taskset -c 6 \
env OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
    OPENBLAS_DYNAMIC=0 \
    KMP_BLOCKTIME=0 \
    KMP_WARNINGS=0 \
    KMP_SETTINGS=0 \
    TF_INTRA_OP_PARALLELISM_THREADS=1 \
    TF_INTER_OP_PARALLELISM_THREADS=1 \
python bench_gemm.py


# ============================================================
# (2) BENCHMARK “1 THREAD SEM PINNING”: core livre + 1 thread
#
# O que este teste mede:
# - Mesmo benchmark GEMM do caso (1), mas sem prender em um único core.
#
# Como está configurado:
# - SEM taskset: o escalonador do SO escolhe o core e pode migrar entre cores.
# - THREADS: 1 thread em OMP/MKL/OpenBLAS/NumExpr (igual ao caso 1).
# - DINÂMICA: desativada (igual ao caso 1) para evitar variação automática.
# - KMP_* e TF_*: iguais ao caso 1 para manter o ambiente controlado.
#
# Por que esse caso é útil:
# - Mostra o “melhor single-thread prático” que o SO consegue entregar,
#   evitando o viés de rodar sempre no CPU 0 (que pode ter mais interrupções).
# ============================================================

env OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
    OPENBLAS_DYNAMIC=0 \
    KMP_BLOCKTIME=0 \
    KMP_WARNINGS=0 \
    KMP_SETTINGS=0 \
    TF_INTRA_OP_PARALLELISM_THREADS=1 \
    TF_INTER_OP_PARALLELISM_THREADS=1 \
    python bench_gemm.py


# ============================================================
# (3) BENCHMARK “CAPACIDADE MÁXIMA: todos os cores + NCPU threads
#
# O que este teste mede:
# - Pico de throughput da BLAS em multi-thread (capacidade máxima de computação),
#   limitado pelo número de CPUs lógicos disponíveis.
#
# Como está configurado:
# - PINNING EM CONJUNTO: restringe o processo ao conjunto 0..NCPU-1 (todos).
#   (não fixa em 1 core; apenas garante que ficará dentro desse conjunto).
# - THREADS: usa NCPU threads em OMP/MKL/OpenBLAS/NumExpr.
# - DINÂMICA: desativada para evitar que a BLAS altere threads durante o teste.
# - KMP_*: reduz overhead e estabiliza o comportamento do runtime (MKL).
# - TF_*: definido por consistência; não é determinante para NumPy.
#
# Observação:
# - Este resultado não é comparável diretamente com o “modo single-core;
#   ele mede a capacidade total do host em paralelo para GEMM.
# ============================================================

NCPU=$(nproc)
CPUSET="0-$((NCPU-1))"

taskset -c "$CPUSET" \
env OMP_NUM_THREADS="$NCPU" \
    MKL_NUM_THREADS="$NCPU" \
    OPENBLAS_NUM_THREADS="$NCPU" \
    NUMEXPR_NUM_THREADS="$NCPU" \
    OMP_DYNAMIC=FALSE \
    MKL_DYNAMIC=FALSE \
    OPENBLAS_DYNAMIC=0 \
    KMP_BLOCKTIME=0 \
    KMP_WARNINGS=0 \
    KMP_SETTINGS=0 \
    TF_INTRA_OP_PARALLELISM_THREADS="$NCPU" \
    TF_INTER_OP_PARALLELISM_THREADS=1 \
    python bench_gemm.py


# 