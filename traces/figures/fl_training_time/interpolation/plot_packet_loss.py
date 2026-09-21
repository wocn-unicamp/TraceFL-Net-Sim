import matplotlib.pyplot as plt
import numpy as np

FS = 11  # font size
ORANGE = "#FFA500"
BLUE = "#6EC6E6"

# Eje x: packet loss = 1 - p  (p = 1.0, 0.95, 0.9, 0.85, 0.8)
loss_labels = ["0", "5", "10", "15", "20"]


def plot(fname, comp, comm, err, legend_loc, ymax):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=200)
    x = np.arange(len(loss_labels))
    ax.bar(x, comp, width=0.8, color=ORANGE, label="Computing")
    ax.bar(x, comm, width=0.8, bottom=comp, color=BLUE, label="Communication",
           yerr=err, capsize=4,
           error_kw={"ecolor": "black", "elinewidth": 1.2, "capthick": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(loss_labels, fontsize=FS)
    ax.tick_params(axis="y", labelsize=FS)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Time (min)", fontsize=FS)
    ax.set_xlabel("Frame Loss Rate (%)", fontsize=FS)
    ax.set_ylim(0, ymax * 1.25)
    ax.legend(loc=legend_loc, fontsize=FS)
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- FEMNIST (valores medidos de la figura original, reordenados por packet loss) ----------
#                   0%     5%     10%    15%    20%
comp_fem = np.array([23.94, 20.42, 17.76, 15.58, 14.48])
total_fem = np.array([40.91, 42.30, 44.12, 46.36, 49.21])
comm_fem = total_fem - comp_fem
err_fem = np.array([0.12, 0.13, 0.13, 0.14, 0.15])
plot("training_time_femnist_loss.png",
     comp_fem, comm_fem, err_fem, legend_loc="upper left", ymax=49.21)

# ---------- Shakespeare (clientes = 10) ----------
total_sha = np.array([29.26, 29.28, 29.26, 29.27, 29.25])
comm_sha = np.array([0.12, 0.12, 0.12, 0.13, 0.13])
comp_sha = total_sha - comm_sha
err_sha = np.array([0.10, 0.11, 0.10, 0.10, 0.11])
plot("training_time_shakespeare_loss.png",
     comp_sha, comm_sha, err_sha, legend_loc="upper left", ymax=38.97)
