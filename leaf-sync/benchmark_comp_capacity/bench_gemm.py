import numpy as np
import time
import os

def bench_gemm(N=2048, dtype=np.float32, repeats=5):
    A = np.random.rand(N, N).astype(dtype)
    B = np.random.rand(N, N).astype(dtype)

    # warm-up
    np.dot(A, B)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np.dot(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best_time = min(times)
    flops = 2 * (N ** 3)
    gflops = flops / best_time / 1e9

    return gflops, best_time

if __name__ == "__main__":
    N = int(os.getenv("N", 2048))
    dtype = os.getenv("DTYPE", "float32")
    dtype = np.float32 if dtype == "float32" else np.float64

    gflops, t = bench_gemm(N=N, dtype=dtype)

    print("================================")
    print(f"N           = {N}")
    print(f"dtype       = {dtype}")
    print(f"time (s)    = {t:.6f}")
    print(f"GFLOP/s     = {gflops:.2f}")
    print("================================")
