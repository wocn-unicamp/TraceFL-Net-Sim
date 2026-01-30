# -*- coding: utf-8 -*-
import time
import os
import csv
import numpy as np
from datetime import datetime

CSV_FILE = "bench_gemm_results.csv"

def bench_gemm(n=4096, dtype=np.float32, warmup=2, iters=5):
    A = np.random.rand(n, n).astype(dtype)
    B = np.random.rand(n, n).astype(dtype)

    # Warm-up
    for _ in range(warmup):
        _ = A @ B

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        times.append(t1 - t0)

    t = min(times)
    flops = 2 * (n ** 3)
    gflops = flops / t / 1e9
    return t, gflops


def append_csv(row: dict, filename: str = CSV_FILE):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Escribe header solo si el archivo no existe
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


if __name__ == "__main__":

    # Variables de entorno relevantes (las que tú ya controlas desde bash)
    env = {
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", ""),
        "MKL_NUM_THREADS": os.getenv("MKL_NUM_THREADS", ""),
        "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS", ""),
        "NUMEXPR_NUM_THREADS": os.getenv("NUMEXPR_NUM_THREADS", ""),
    }

    for dtype in (np.float32, np.float64):
        n = 4096 if dtype == np.float32 else 3072
        t, g = bench_gemm(n=n, dtype=dtype)

        # Salida estándar (no la rompemos)
        print(f"dtype={dtype.__name__}  best_time={t:.4f}s  approx_GFLOP/s={g:.2f}")

        # Fila CSV
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dtype": dtype.__name__,
            "n": n,
            "best_time_s": round(t, 6),
            "gflops": round(g, 2),
            **env,
        }

        append_csv(row)
