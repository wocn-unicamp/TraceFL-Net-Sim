import time
import numpy as np

def bench_gemm(n=4096, dtype=np.float32, warmup=2, iters=5):
    # A y B aleatorias
    A = np.random.rand(n, n).astype(dtype)
    B = np.random.rand(n, n).astype(dtype)

    # Warm-up (para “calentar” caches y JIT interno de BLAS)
    for _ in range(warmup):
        _ = A @ B

    # Medición
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        C = A @ B
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t = min(times)  # mejor tiempo (menos ruido)
    # FLOPs de GEMM: ~ 2 * n^3 (multiplicar+sumar)
    flops = 2 * (n ** 3)
    gflops = flops / t / 1e9
    return t, gflops

if __name__ == "__main__":
    for dtype in (np.float32, np.float64):
        t, g = bench_gemm(n=4096 if dtype==np.float32 else 3072, dtype=dtype)
        print(f"dtype={dtype.__name__}  best_time={t:.4f}s  approx_GFLOP/s={g:.2f}")
