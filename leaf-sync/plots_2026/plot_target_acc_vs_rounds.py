import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================== Config =====================
folder_path = "../results/stat/fine/"
file_bases = [
    "stat_metrics_fedavg_c_64_e_1",
    "stat_metrics_fedavg_c_64_e_2",
    "stat_metrics_fedavg_c_64_e_3",
    "stat_metrics_fedavg_c_64_e_4",
    "stat_metrics_fedavg_c_64_e_5",
]

TARGETS = [65.0, 70.0, 74]  # in %

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

grid_alpha = 0.3

# ===================== (NEW) helper to save CSV =====================
def save_rounds_to_targets_csv(Es, targets, rounds_matrix, out_dir, filename="rounds_to_reach_targets_c64.csv"):
    data = {"E": Es.tolist()}
    for j, t in enumerate(targets):
        data[f"round_to_{int(t)}"] = rounds_matrix[:, j].tolist()
    out_path = os.path.join(out_dir, filename)
    pd.DataFrame(data).to_csv(out_path, index=False)
    return out_path

# ===================== Compute rounds to target =====================
Es = []
rounds_matrix = []  # rows: E, cols: targets

for base in file_bases:
    path = os.path.join(folder_path, base + ".csv")
    df = pd.read_csv(path)

    df_test = df[df["set"] == "test"].copy()
    df_test["round_number"] = pd.to_numeric(df_test["round_number"], errors="coerce")
    df_test["accuracy"] = pd.to_numeric(df_test["accuracy"], errors="coerce")
    df_test = df_test.dropna(subset=["round_number", "accuracy"])

    acc_by_round = df_test.groupby("round_number")["accuracy"].mean().sort_index()

    # convert to % if in [0,1]
    if acc_by_round.max() <= 1.001:
        acc_by_round = acc_by_round * 100.0

    # extract E from filename
    m = re.search(r"e_(\d+)", base)
    E = int(m.group(1)) if m else len(Es) + 1
    Es.append(E)

    row = []
    for t in TARGETS:
        hit = acc_by_round[acc_by_round >= t]
        first_round = float(hit.index[0]) if len(hit) > 0 else np.nan
        row.append(first_round)
    rounds_matrix.append(row)

Es = np.array(Es)
rounds_matrix = np.array(rounds_matrix)  # shape: (len(Es), len(TARGETS))

# ===================== Plot (grouped bars) =====================
# ===================== Plot (grouped bars) =====================
x = np.arange(len(Es))
bar_w = 0.22

fig, ax = plt.subplots(figsize=(9, 5))

offsets = (np.arange(len(TARGETS)) - (len(TARGETS) - 1) / 2.0) * bar_w

bar_containers = []
for j, t in enumerate(TARGETS):
    bc = ax.bar(x + offsets[j], rounds_matrix[:, j], width=bar_w, label=f"{t:.0f}%")
    bar_containers.append(bc)

# (NEW) Dar espacio arriba para que no se corten las etiquetas
y_data_max = np.nanmax(rounds_matrix) if np.any(np.isfinite(rounds_matrix)) else 0.0
ax.set_ylim(0, y_data_max * 1.12 + 1.0)
y_max = ax.get_ylim()[1]

# (NEW) Anotar barras
def annotate_bars(bar_container, values):
    for rect, v in zip(bar_container, values):
        if not np.isfinite(v):
            continue

        h = rect.get_height()

        # siempre arriba (con margen y sin salirse del eje)
        y = min(h + max(1.0, 0.01 * y_max), y_max - max(1.0, 0.02 * y_max))
        va = "bottom"

        # formateo: entero si aplica, si no 1 decimal
        label = f"{int(v)}" if float(v).is_integer() else f"{v:.1f}"

        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            y,
            label,
            ha="center",
            va=va,
        )

for j, bc in enumerate(bar_containers):
    annotate_bars(bc, rounds_matrix[:, j])

ax.set_title("Number of training rounds to reach a target accuracy")
ax.set_xlabel("Local epochs (E)")
ax.set_ylabel("Rounds")
ax.set_xticks(x)
ax.set_xticklabels([str(e) for e in Es])
ax.grid(True, axis="y", alpha=grid_alpha)
ax.legend(title="Target accuracy", ncol=len(TARGETS))
fig.tight_layout()

out_dir = os.path.join("figures", "acc_target")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "rounds_to_reach_targets_c64.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================== (NEW) save CSV in same folder as image =====================
csv_path = save_rounds_to_targets_csv(Es, TARGETS, rounds_matrix, out_dir)

print("Saved:", out_path)
print("Saved CSV:", csv_path)
print("E values:", Es.tolist())
print("Rounds to targets (cols = 65,70,74):")
print(rounds_matrix)
