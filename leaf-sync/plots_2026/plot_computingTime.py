import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Funções simples (reuso)
# =========================
def plot_cdf_group_from_column(
    folder: str,
    out_dir: str,
    c: int,
    epochs,
    column: str,
    transform_fn,          # função: (df) -> array 1D com os valores de x
    title: str,
    xlabel: str,
    out_filename: str
):
    """
    Lê um CSV por epoch (e), extrai uma série x (via transform_fn), e plota CDF em grupo.
    - Sem validações extras: assume que os arquivos e colunas existem.
    - Comentários PT-BR, estilo direto.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(7, 5))
    
    # Name of the plot
    print(f"\nUsing '{column}' to plot CDF...")  

    for e in epochs:
        path = os.path.join(folder, f"sys_metrics_fedavg_c_{c}_e_{e}.csv")
        df = pd.read_csv(path)

        # Extrai e transforma os dados (ex: GFLOP, tempo estimado, etc.)
        x = np.sort(transform_fn(df))
        y = np.arange(1, len(x) + 1) / len(x)

        # max value in x
        print(f"EPOCH {e} - MAX VALUE in {column}: {x[-1]}")
        # Percentils
        print(f"Percentiles:  99 {np.percentile(x, 99)}  99.5 {np.percentile(x, 99.5)}  99.9 {np.percentile(x, 99.9)}")

        plt.plot(x, y, linewidth=2, label=f"e={e}")

    # Parte “repetida” encapsulada
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Clients (%)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_filename), dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# Uso (3 gráficos)
# =========================
FOLDER = "../results/sys/"
OUT = "figures/computingTime"
EPOCHS = range(1, 5)
C = 64
FLOPS_PER_SEC = 64e9  # 64 GFLOP/s
REAL_FLOPS_PER_SEC = 100e9  # 100 GFLOP/s medido no benchmark (modo single-core)



# 1) CDF: local_computations (GFLOP)
plot_cdf_group_from_column(
    folder=FOLDER,
    out_dir=OUT,
    c=C,
    epochs=EPOCHS,
    column="local_computations",
    transform_fn=lambda df: df["local_computations"].dropna().to_numpy() / 1e9,
    title=f" CDF of computing demand in GFLOPs",
    xlabel="Computing Demand per round (GFLOPs)",
    out_filename=f"computational_demand_gflop_c_{C}.png"
)

# 2) CDF: tempo estimado = FLOPs / 64 GFLOP/s
plot_cdf_group_from_column(
    folder=FOLDER,
    out_dir=OUT,
    c=C,
    epochs=EPOCHS,
    column="local_computations",
    transform_fn=lambda df: df["local_computations"].dropna().to_numpy() / FLOPS_PER_SEC,
    title=f"CDF of estimated computing time (With 64 GFLOPs/sec)",
    xlabel="Estimated Computing Time (s)",
    out_filename=f"estimated_computing_time_c_{C}.png"
)

# 3) CDF: computingTime real (coluna do CSV)
plot_cdf_group_from_column(
    folder=FOLDER,
    out_dir=OUT,
    c=C,
    epochs=EPOCHS,
    column="computingTime",
    transform_fn=lambda df: df["computingTime"].dropna().to_numpy(),
    title=f"CDF of real computing time (seconds)",
    xlabel="Real Computing Time (s)",
    out_filename=f"real_computing_time_c_{C}.png"
)

# 4) CDF: Estimated local computations (GFLOP)
plot_cdf_group_from_column(
    folder=FOLDER,
    out_dir=OUT,
    c=C,
    epochs=EPOCHS,
    column="computingTime",
    transform_fn=lambda df: df["computingTime"].dropna().to_numpy() * REAL_FLOPS_PER_SEC,
    title=f"CDF of estimated computing demand (GFLOPs)",
    xlabel="Estimated Computing Demand per round in GFLOP (With 100 GFLOPs/sec)",
    out_filename=f"estimated_computational_demand_gflop_c_{C}.png"
)