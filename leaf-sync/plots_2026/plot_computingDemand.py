import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Estilo global
# =========================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

linewidth = 1.3      # linhas mais finas (não “pesadas”)
grid_alpha = 0.3

# =========================
# Config
# =========================
SIM_TYPE = "serial_lowcap"  # "paralelo" ou "serial" ou "serial_lowcap"
FOLDER = f"../results/sys/fine_{SIM_TYPE}/"
OUT_DIR = f"figures/computingDemand/{SIM_TYPE}"

EPOCHS = range(1, 6)
C = 64

# Pontos (para não poluir)
MAX_POINTS_PER_E = 350   # subamostra por epoch
X_JITTER = 0.12
RNG_SEED = 7

os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Carrega dados (mesmo trace)
# =========================
data = []
labels = []

for e in EPOCHS:
    # Lê o CSV do epoch E
    path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{e}_time.csv")
    df = pd.read_csv(path)

    # FLOPs -> GFLOPs
    gflops = df["computingDemand"].dropna().to_numpy() / 1e9

    data.append(gflops)
    labels.append(f"{e}")

# =========================
# Plot: boxplot + pontos (jitter)
# =========================
plt.figure(figsize=(7.5, 5))
x = np.arange(1, len(data) + 1)

# Boxplot (sem outliers, pois os pontos já mostram tudo)
bp = plt.boxplot(
    data,
    positions=x,
    widths=0.55,
    showfliers=False,
    patch_artist=True,
    boxprops=dict(linewidth=linewidth),
    whiskerprops=dict(linewidth=linewidth),
    capprops=dict(linewidth=linewidth),
    medianprops=dict(linewidth=linewidth + 0.4),
)

# Deixa as caixas discretas (sem cor forte)
for box in bp["boxes"]:
    box.set_alpha(0.20)

# Pontos com jitter no eixo x (subamostrados)
rng = np.random.default_rng(RNG_SEED)
for i, y in enumerate(data, start=1):
    if len(y) > MAX_POINTS_PER_E:
        idx = rng.choice(len(y), size=MAX_POINTS_PER_E, replace=False)
        y_plot = y[idx]
    else:
        y_plot = y

    x_plot = i + rng.uniform(-X_JITTER, X_JITTER, size=len(y_plot))
    plt.scatter(x_plot, y_plot, s=12, alpha=0.20, linewidths=0)

# Acabamento
plt.title("Computing Demand per Round vs. Local Epochs")
plt.xlabel("Local epochs (E)")
plt.ylabel("Computing demand per round (GFLOPs)")
plt.xticks(x, labels)
plt.ylim(bottom=0)
plt.grid(True, axis="y", alpha=grid_alpha)
plt.tight_layout()

out_path = os.path.join(OUT_DIR, f"box_scatter_computing_demand_gflops_c_{C}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path}")
