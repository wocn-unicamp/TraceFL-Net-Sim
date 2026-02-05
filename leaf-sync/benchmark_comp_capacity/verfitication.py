import os
import numpy as np

print("NumPy:", np.__version__)
print("OMP_NUM_THREADS:", os.getenv("OMP_NUM_THREADS"))
print("MKL_NUM_THREADS:", os.getenv("MKL_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS:", os.getenv("OPENBLAS_NUM_THREADS"))
print("NUMEXPR_NUM_THREADS:", os.getenv("NUMEXPR_NUM_THREADS"))

# Afinidad visible desde Python
try:
    aff = sorted(os.sched_getaffinity(0))
    print("CPU affinity (sched_getaffinity):", f"{aff[0]}-{aff[-1]}" if aff else aff)
except Exception as e:
    print("CPU affinity not available:", e)

from threadpoolctl import threadpool_info
blas = [d for d in threadpool_info() if d.get("user_api") == "blas"]
print("BLAS backends and effective threads:")
for d in blas:
    print(f" - {d.get('internal_api')} : num_threads={d.get('num_threads')}  version={d.get('version')}")
