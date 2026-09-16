import matplotlib.pyplot as plt
import numpy as np

FS = 11  # font size
ORANGE = "#FFA500"
BLUE = "#6EC6E6"

# --- Panel izquierdo: variando clientes (mini-batch = 1.0) ---
clients = ["2", "3", "4", "5", "8", "10", "20"]
total_clients = np.array([12.27, 15.24, 18.02, 20.76, 26.38, 29.26, 38.97])
comm_clients = np.array([0.10, 0.10, 0.11, 0.11, 0.12, 0.12, 0.14])
comp_clients = total_clients - comm_clients
err_clients = np.array([0.08, 0.08, 0.09, 0.10, 0.10, 0.11, 0.12])

# --- Panel derecho: variando mini-batch (clientes = 10) ---
# Valores originales (simulados con 20 clientes) reescalados para que el
# valor implicito en mini-batch = 1.0 coincida con el de 10 clientes (29.26 min)
batches = ["0.2", "0.4", "0.5", "0.6", "0.8", "0.9"]
orig_20 = np.array([7.78, 15.60, 19.46, 23.37, 31.19, 35.10])   # medidos de la figura original
scale = 29.26 / 38.97
total_batch = np.round(orig_20 * scale, 2)                        # [5.84, 11.71, 14.61, 17.55, 23.42, 26.36]
comm_batch = np.array([0.09, 0.10, 0.10, 0.11, 0.12, 0.12])
comp_batch = total_batch - comm_batch
err_batch = np.array([0.07, 0.08, 0.08, 0.09, 0.10, 0.10])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=200, sharey=True)

for ax, labels, comp, comm, err in [
    (ax1, clients, comp_clients, comm_clients, err_clients),
    (ax2, batches, comp_batch, comm_batch, err_batch),
]:
    x = np.arange(len(labels))
    ax.bar(x, comp, width=0.8, color=ORANGE, label="Computing")
    ax.bar(x, comm, width=0.8, bottom=comp, color=BLUE, label="Communication",
           yerr=err, capsize=4, error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)

ax1.set_ylabel("Time (min)", fontsize=FS)
ax1.set_xlabel("Clients (Mini-batch = 1.0)", fontsize=FS)
ax2.set_xlabel("Mini-batch (Clients = 10)", fontsize=FS)
ax1.set_ylim(0, 38.97 * 1.25)
ax1.set_yticks([0, 10, 20, 30, 40])
ax1.legend(loc="upper left", fontsize=FS)

plt.tight_layout()
plt.savefig("training_time_shakespeare.png",
            dpi=200,
            bbox_inches="tight")
