import os
import pandas as pd
import matplotlib.pyplot as plt

# Pasta e lista de arquivos-base (com ou sem .csv)
folder_path = "../results/stat/"


alg = "minibatch"  # Pode ser "fedavg" ou "minibatch"


file_base = "stat_metrics_" + alg + "_"

file_bases = [
    "stat_metrics_minibatch_c_5_mb_1",
    "stat_metrics_minibatch_c_10_mb_1",
    "stat_metrics_minibatch_c_30_mb_1"
]

# Linestyles diferentes para cada arquivo (cores ficam a cargo do matplotlib)
linestyles = ["-", "--", ":", "-."]

def ensure_csv(path_base: str) -> str:
    return path_base if path_base.endswith(".csv") else path_base + ".csv"

def short_label(name: str) -> str:
    # Ex.: "stat_metrics_minibatch_c_5_mb_1.csv" -> "c_5_mb_1"
    base = os.path.basename(name).replace(".csv", "")
    return base.replace("stat_metrics_minibatch_", "")

# 4 subplots: [0,0]=Acc Train | [0,1]=Acc Test | [1,0]=Loss Train | [1,1]=Loss Test
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

for i, base in enumerate(file_bases):
    file_path = os.path.join(folder_path, ensure_csv(base))
    if not os.path.exists(file_path):
        print(f"[aviso] arquivo não encontrado: {file_path} — ignorando.")
        continue

    df = pd.read_csv(file_path)

    df_train = df[df["set"] == "train"]
    df_test  = df[df["set"] == "test"]

    # Métricas por round (ordenadas)
    acc_train = df_train.groupby("round_number")["accuracy"].mean().sort_index()
    acc_test  = df_test.groupby("round_number")["accuracy"].mean().sort_index()
    loss_train = df_train.groupby("round_number")["loss"].mean().sort_index()
    loss_test  = df_test.groupby("round_number")["loss"].mean().sort_index()

    label_tag = short_label(file_path)
    ls = linestyles[i % len(linestyles)]

    # ----- Accuracy -----
    if not acc_train.empty:
        axs[0, 0].plot(acc_train.index, acc_train.values,
                       linestyle=ls, marker='s', label=f"{label_tag} — Train")
    if not acc_test.empty:
        axs[0, 1].plot(acc_test.index, acc_test.values,
                       linestyle=ls, marker='o', label=f"{label_tag} — Test")

    # ----- Loss -----
    if not loss_train.empty:
        axs[1, 0].plot(loss_train.index, loss_train.values,
                       linestyle=ls, marker='s', label=f"{label_tag} — Train")
    if not loss_test.empty:
        axs[1, 1].plot(loss_test.index, loss_test.values,
                       linestyle=ls, marker='o', label=f"{label_tag} — Test")

# Títulos, rótulos e grades
axs[0, 0].set_title("Accuracy — Train")
axs[0, 1].set_title("Accuracy — Test")
axs[1, 0].set_title("Loss — Train")
axs[1, 1].set_title("Loss — Test")

axs[1, 0].set_xlabel("Rounds")
axs[1, 1].set_xlabel("Rounds")
axs[0, 0].set_ylabel("Accuracy")
axs[1, 0].set_ylabel("Loss")

# ylim and xlim for accuracy plots
axs[0, 0].set_ylim(0, 0.8)
axs[0, 1].set_ylim(0, 0.8)
axs[0, 0].set_xlim(0, 500)
axs[0, 1].set_xlim(0, 500)

for ax in axs.ravel():
    ax.grid(True)
    ax.legend(ncol=1)

plt.tight_layout()

# Salvar figura
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/plot_acc_loss_vs_rounds_fedavg_train_test_split.png", dpi=150)

plt.show()
