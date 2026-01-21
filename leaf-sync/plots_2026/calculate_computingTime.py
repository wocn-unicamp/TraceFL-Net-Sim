import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR = "../results_2026/sys/"   # carpeta donde están los CSV
OUT_DIR = "trace_sim"              # carpeta de salida (CSVs con computingTime)
os.makedirs(OUT_DIR, exist_ok=True)

FIG_DIR = "figures/cdf_computingTime"
os.makedirs(FIG_DIR, exist_ok=True)

C_FIXED = 64
E_VALUES = [1, 2, 3]

# Capacidad del dispositivo: 64 GFLOP/s = 64e9 FLOP/s
DEVICE_FLOPS_PER_SEC = 64 * 1e9

# =========================
# MAIN
# =========================
for e in E_VALUES:
    in_name = f"sys_metrics_fedavg_c_{C_FIXED}_e_{e}.csv"
    in_path = os.path.join(DATA_DIR, in_name)

    if not os.path.exists(in_path):
        print(f"[warn] No existe: {in_path} (saltando)")
        continue

    # CSV sin header
    df = pd.read_csv(in_path, header=None)

    # Ajusta estos nombres si tu logger usa otro orden/semántica
    df.columns = [
        "client_id", "round", "aux", "num_samples",
        "phase", "bytes_up", "bytes_down", "flops"
    ]

    # Tipos numéricos básicos
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df["flops"] = pd.to_numeric(df["flops"], errors="coerce")
    df = df.dropna(subset=["round", "flops"]).copy()
    df["round"] = df["round"].astype(int)

    # ===== computingTime (segundos) por cliente por rodada =====
    df["computingTime"] = df["flops"] / DEVICE_FLOPS_PER_SEC

    # Guarda CSV con la nueva columna (sin agregar c ni e)
    out_name = f"sys_metrics_fedavg_c_{C_FIXED}_e_{e}_with_computingTime.csv"
    out_path = os.path.join(OUT_DIR, out_name)
    df.to_csv(out_path, index=False)
    print(f"[ok] Guardado CSV: {out_path}")

    # ===== CDF de computingTime =====
    x = df["computingTime"].dropna().to_numpy()
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, linewidth=2)
    plt.title(f"CDF of Computing Time (c={C_FIXED}, e={e})")
    plt.xlabel("computingTime (s)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(FIG_DIR, f"cdf_computingTime_c_{C_FIXED}_e_{e}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[ok] Guardada figura: {fig_path}")
