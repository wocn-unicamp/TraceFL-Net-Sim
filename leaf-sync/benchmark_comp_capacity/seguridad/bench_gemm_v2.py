#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEMM micro-benchmark for NumPy/BLAS.

Designed to be driven by run_bench_gemm.sh without CLI parameters.
It records results to a CSV with a stable, minimal schema and captures
the *effective* BLAS thread count via threadpoolctl to validate that
the requested threading settings are actually being applied.
"""

from __future__ import annotations

import csv
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except Exception:  # pragma: no cover
    threadpool_info = None  # type: ignore
    threadpool_limits = None  # type: ignore


# -------------------------
# Defaults (can be overridden via env from the bash driver)
# -------------------------
DEFAULT_N_F32 = int(os.getenv("BENCH_N_F32", "4096"))
DEFAULT_N_F64 = int(os.getenv("BENCH_N_F64", "3072"))
DEFAULT_WARMUP = int(os.getenv("BENCH_WARMUP", "2"))
DEFAULT_ITERS = int(os.getenv("BENCH_ITERS", "5"))
DEFAULT_SEED = int(os.getenv("BENCH_SEED", "12345"))

# CSV output (can be overridden via env from the bash driver)
DEFAULT_CSV_NAME = "bench_gemm_results.csv"
CSV_FILE = os.getenv("CSV_FILE", "")  # if empty, use script-local default


CSV_FIELDS = [
    "timestamp",
    "run_id",
    "scenario",
    "pinning",
    "cpu_set",
    "threads_req",
    "threads_eff",
    "blas_backend",
    "dtype",
    "n",
    "best_time_s",
    "gflops",
    "python",
    "numpy",
    "host",
    "cpu_model",
]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_csv_path() -> Path:
    return _script_dir() / DEFAULT_CSV_NAME


def _csv_path() -> Path:
    if CSV_FILE.strip():
        return Path(CSV_FILE).expanduser().resolve()
    return _default_csv_path()


def _read_first_line(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return f.readline().strip()
    except FileNotFoundError:
        return ""


def _ensure_csv_schema(path: Path) -> None:
    """Ensures CSV exists with the expected header.
    If a different header exists, it is backed up and a fresh file is created.
    """
    header = ",".join(CSV_FIELDS)
    first = _read_first_line(path)
    if not first:
        # Create new file with header
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(header + "\n")
        return

    if first != header:
        # Backup and recreate
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_suffix(path.suffix + f".bak_{ts}")
        path.rename(backup)
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write(header + "\n")


def _append_csv_row(path: Path, row: dict) -> None:
    _ensure_csv_schema(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        # fill missing fields
        for k in CSV_FIELDS:
            row.setdefault(k, "")
        writer.writerow(row)


def _cpu_model() -> str:
    # Linux: /proc/cpuinfo is the most reliable
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    # Fallbacks
    return platform.processor() or platform.uname().processor or "unknown"


def _compress_affinity(cpus: Iterable[int]) -> str:
    lst = sorted(set(int(x) for x in cpus))
    if not lst:
        return ""
    ranges = []
    start = prev = lst[0]
    for x in lst[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append((start, prev))
        start = prev = x
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(parts)


def _current_affinity() -> str:
    try:
        cpus = os.sched_getaffinity(0)  # type: ignore[attr-defined]
        return _compress_affinity(cpus)
    except Exception:
        return ""


def _blas_backend_and_threads() -> Tuple[str, str]:
    """Returns (backend, effective_threads) based on threadpoolctl, if available."""
    if threadpool_info is None:
        return "", ""

    try:
        infos = threadpool_info()
    except Exception:
        return "", ""

    blas_infos = [d for d in infos if d.get("user_api") == "blas"]
    if not blas_infos:
        return "", ""

    # Pick unique backends (e.g., openblas, mkl, blis) and the max thread count reported
    backends = []
    max_thr = None
    for d in blas_infos:
        api = str(d.get("internal_api", "")).strip()
        if api and api not in backends:
            backends.append(api)
        thr = d.get("num_threads", None)
        if isinstance(thr, int):
            max_thr = thr if max_thr is None else max(max_thr, thr)

    backend_str = "+".join(backends)
    thr_str = str(max_thr) if max_thr is not None else ""
    return backend_str, thr_str


def _requested_threads() -> Optional[int]:
    # Prefer explicit value from the driver; fallback to common env vars.
    for key in ("BENCH_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        v = os.getenv(key, "").strip()
        if v:
            try:
                t = int(v)
                if t > 0:
                    return t
            except ValueError:
                pass
    return None


def bench_gemm(n: int, dtype: np.dtype, warmup: int, iters: int, seed: int, force_threads: Optional[int]) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    A = rng.random((n, n), dtype=dtype)
    B = rng.random((n, n), dtype=dtype)

    def _matmul():
        _ = A @ B  # noqa: F841

    # Warm-up
    for _ in range(warmup):
        if threadpool_limits is not None and force_threads is not None:
            with threadpool_limits(limits=force_threads, user_api="blas"):
                _matmul()
        else:
            _matmul()

    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        if threadpool_limits is not None and force_threads is not None:
            with threadpool_limits(limits=force_threads, user_api="blas"):
                _matmul()
        else:
            _matmul()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    best = min(times)
    flops = 2.0 * (n ** 3)
    gflops = flops / best / 1e9
    return best, gflops


def main() -> int:
    run_id = os.getenv("BENCH_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    scenario = os.getenv("BENCH_SCENARIO", "")
    pinning = os.getenv("BENCH_PINNING", "")
    cpu_set = os.getenv("BENCH_CPUSET", "")

    warmup = int(os.getenv("BENCH_WARMUP", str(DEFAULT_WARMUP)))
    iters = int(os.getenv("BENCH_ITERS", str(DEFAULT_ITERS)))
    seed = int(os.getenv("BENCH_SEED", str(DEFAULT_SEED)))

    thr_req = _requested_threads()
    # Enforce BLAS threads with threadpoolctl to avoid cases where env is ignored.
    force_threads = thr_req

    host = platform.node()
    cpu_model = _cpu_model()
    pyver = platform.python_version()
    npver = np.__version__

    out_csv = _csv_path()

    for dtype in (np.float32, np.float64):
        n = DEFAULT_N_F32 if dtype == np.float32 else DEFAULT_N_F64
        best, gflops = bench_gemm(
            n=n,
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            seed=seed,
            force_threads=force_threads,
        )

        backend, thr_eff = _blas_backend_and_threads()
        # If threadpoolctl reports nothing, fallback to affinity/env-based info.
        if not thr_eff and thr_req is not None:
            thr_eff = str(thr_req)

        # Human-readable stdout (kept stable for the bash driver)
        print(f"dtype={dtype.__name__}  best_time={best:.4f}s  approx_GFLOP/s={gflops:.2f}")

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": run_id,
            "scenario": scenario,
            "pinning": pinning,
            "cpu_set": cpu_set,
            "threads_req": "" if thr_req is None else str(thr_req),
            "threads_eff": thr_eff,
            "blas_backend": backend,
            "dtype": dtype.__name__,
            "n": int(n),
            "best_time_s": f"{best:.6f}",
            "gflops": f"{gflops:.2f}",
            "python": pyver,
            "numpy": npver,
            "host": host,
            "cpu_model": cpu_model,
        }

        _append_csv_row(out_csv, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
