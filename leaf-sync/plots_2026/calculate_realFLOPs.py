import pandas as pd
import numpy as np

for epoch in range(1, 4):  

    folder_path = "../results/sys/"

    df = pd.read_csv(folder_path + f"sys_metrics_fedavg_c_10_e_{epoch}.csv")

    # Evitar divisiones inválidas
    df = df[(df["computingTime"] > 0) & (df["local_computations"] > 0)].copy()

    # Throughput efectivo - ¿A qué velocidad real (GFLOP/s) se ejecutó este cliente?
    df["effective_gflops"] = df["local_computations"] / df["computingTime"] / 1e9

    # Tiempo normalizado - ¿Cuántos segundos cuesta ejecutar 1 GFLOP?
    df["sec_per_gflop"] = df["computingTime"] / (df["local_computations"] / 1e9)

    print(f"\n\n                  ========= Epoch {epoch} =========")
    
    # Estadísticas básicas
    mean = df["effective_gflops"].mean()
    std = df["effective_gflops"].std()
    ci95 = 1.96 * std / np.sqrt(len(df))

    print(f"Effective GFLOPS: Mean = {mean:.4f}, Std = {std:.4f}, CI95%  = ±{ci95:.4f}")
    print(df[["client_id", "round_number", "effective_gflops", "sec_per_gflop"]].head(10))
