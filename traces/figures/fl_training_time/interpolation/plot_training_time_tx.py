import matplotlib.pyplot as plt
import numpy as np

FS = 11  # font size
ORANGE = "#FFA500"
BLUE = "#6EC6E6"

# Probabilidad de transmision exitosa de frame (clientes = 10, mini-batch = 1.0)
# El tiempo no depende de la probabilidad: todas las barras al valor de 10 clientes (~29.26 min)
probs = ["0.8", "0.85", "0.9", "0.95", "1.0"]
total = np.array([29.25, 29.27, 29.26, 29.28, 29.26])
comm = np.array([0.13, 0.13, 0.12, 0.12, 0.12])
comp = total - comm
err = np.array([0.10, 0.11, 0.10, 0.10, 0.11])

fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=200)

x = np.arange(len(probs))
ax.bar(x, comp, width=0.8, color=ORANGE, label="Computing")
ax.bar(x, comm, width=0.8, bottom=comp, color=BLUE, label="Communication",
       yerr=err, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2})
ax.set_xticks(x)
ax.set_xticklabels(probs, fontsize=FS)
ax.tick_params(axis="y", labelsize=FS)
ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
ax.set_axisbelow(True)

ax.set_ylabel("Time (min)", fontsize=FS)
ax.set_xlabel("Probability of successful\nframe transmission", fontsize=FS)
ax.set_ylim(0, 38.97 * 1.25)   # mismo eje que la figura original
ax.set_yticks([0, 10, 20, 30, 40])
ax.legend(loc="upper left", fontsize=FS)

plt.tight_layout()
plt.savefig("training_time_shakespeare_tx.png",
            dpi=200,
            bbox_inches="tight")
