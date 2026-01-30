import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = "../results/sys/"
EPOCHS = range(1, 4)
OUT = "figures/real_computingTime"
os.makedirs(OUT, exist_ok=True)

plt.figure(figsize=(7, 5))

for epoch in EPOCHS:
    path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_10_e_{epoch}.csv")
    df = pd.read_csv(path)
    x = np.sort(df["computingTime"].dropna().to_numpy())
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, linewidth=2, label=f"e={epoch}")

plt.title("CDF of computingTime (c=10)")
plt.xlabel("computingTime (s)")
plt.ylabel("CDF")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "cdf_computingTime_c_10_ALL_e.png"), dpi=150, bbox_inches="tight")
plt.close()
