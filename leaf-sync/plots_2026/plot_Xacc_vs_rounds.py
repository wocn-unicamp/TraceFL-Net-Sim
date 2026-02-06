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
x = np.arange(len(Es))
bar_w = 0.22

plt.figure(figsize=(9, 5))

offsets = (np.arange(len(TARGETS)) - (len(TARGETS) - 1) / 2.0) * bar_w
for j, t in enumerate(TARGETS):
    plt.bar(x + offsets[j], rounds_matrix[:, j], width=bar_w, label=f"{t:.0f}%")

plt.title("Rounds to Reach a Target Accuracy (C = 64)")
plt.xlabel("Local epochs (E)")
plt.ylabel("Rounds")
plt.xticks(x, [str(e) for e in Es])
plt.grid(True, axis="y", alpha=grid_alpha)
plt.legend(title="Target accuracy", ncol=len(TARGETS))
plt.tight_layout()

out_dir = os.path.join("figures", "acc_target")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rounds_to_reach_targets_c64.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", out_path)
print("E values:", Es.tolist())
print("Rounds to targets (cols = 65,70,74):")
print(rounds_matrix)
