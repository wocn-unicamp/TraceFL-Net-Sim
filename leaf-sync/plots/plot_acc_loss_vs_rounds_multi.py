import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Pasta e lista de arquivos-base (com ou sem .csv)
folder_path = "../results/stat/"

# file_bases = [
#     "stat_metrics_shakespeare_fedavg_c_2_e_1",
#     "stat_metrics_shakespeare_fedavg_c_3_e_1",
#     "stat_metrics_shakespeare_fedavg_c_4_e_1",
#     "stat_metrics_shakespeare_fedavg_c_5_e_1",
#     "stat_metrics_shakespeare_fedavg_c_8_e_1",
# ]


# file_bases = [
#     "stat_metrics_fedavg_c_50_e_1",
#     "stat_metrics_fedavg_c_30_e_1",
#     "stat_metrics_fedavg_c_20_e_1",
#     "stat_metrics_fedavg_c_10_e_1",
#     "stat_metrics_fedavg_c_5_e_1",
#     "stat_metrics_fedavg_c_3_e_1"
# ]


file_bases = [
    "stat_metrics_minibatch_c_20_mb_1",
    "stat_metrics_minibatch_c_20_mb_0.9",
    "stat_metrics_minibatch_c_20_mb_0.8",
    "stat_metrics_minibatch_c_20_mb_0.6",
    "stat_metrics_minibatch_c_20_mb_0.5",
    "stat_metrics_minibatch_c_20_mb_0.4",
    "stat_metrics_minibatch_c_20_mb_0.2",
]

name_figure = file_bases[0].split("_")[2]  # ex.: "c_5_e_1"


# Linestyles diferentes para cada arquivo (cores ficam a cargo do matplotlib)
linestyles = ["-", "--", ":", "-."]

def ensure_csv(path_base: str) -> str:
    return path_base if path_base.endswith(".csv") else path_base + ".csv"

def short_label(name: str) -> str:
    """
    Ex.: "stat_metrics_minibatch_c_5_mb_1.csv" -> "c_5_mb_1"
         "stat_metrics_fedavg_c_10_e_1.csv"    -> "c_10_e_1"
         "stat_metrics_shakespeare_fedavg_c_3_e_1.csv" -> "c_3_e_1"
    """
    base = os.path.basename(name).replace(".csv", "")
    base = re.sub(r"^stat_metrics_(shakespeare_)?(minibatch_|fedavg_)?", "", base)
    return base

# 4 subplots: [0,0]=Acc Train | [0,1]=Acc Test | [1,0]=Loss Train | [1,1]=Loss Test
fig, axs = plt.subplots(2, 2, figsize=(12, 9), sharex=True)

expected_cols = {"set", "round_number", "accuracy", "loss"}

for i, base in enumerate(file_bases):
    file_path = os.path.join(folder_path, ensure_csv(base))
    if not os.path.exists(file_path):
        print(f"[aviso] arquivo não encontrado: {file_path} — ignorando.")
        continue

    df = pd.read_csv(file_path)

    # Checagem mínima de colunas
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[aviso] faltam colunas {missing} em {file_path} — ignorando.")
        continue

    # Garantir tipos numéricos para plot
    df["round_number"] = pd.to_numeric(df["round_number"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
    df = df.dropna(subset=["round_number"])  # descarta rounds inválidos

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
                       linestyle=ls, marker='None', label=f"{label_tag} — Train")
    else:
        print(f"[aviso] sem dados de train/accuracy em {file_path}.")
    if not acc_test.empty:
        axs[0, 1].plot(acc_test.index, acc_test.values,
                       linestyle=ls, marker='None', label=f"{label_tag} — Test")
    else:
        print(f"[aviso] sem dados de test/accuracy em {file_path}.")

    # ----- Loss -----
    if not loss_train.empty:
        axs[1, 0].plot(loss_train.index, loss_train.values,
                       linestyle=ls, marker='None', label=f"{label_tag} — Train")
    else:
        print(f"[aviso] sem dados de train/loss em {file_path}.")
    if not loss_test.empty:
        axs[1, 1].plot(loss_test.index, loss_test.values,
                       linestyle=ls, marker='None', label=f"{label_tag} — Test")
    else:
        print(f"[aviso] sem dados de test/loss em {file_path}.")

# Títulos, rótulos e grades
axs[0, 0].set_title("Accuracy — Train")
axs[0, 1].set_title("Accuracy — Test")
axs[1, 0].set_title("Loss — Train")
axs[1, 1].set_title("Loss — Test")

axs[1, 0].set_xlabel("Rounds")
axs[1, 1].set_xlabel("Rounds")
axs[0, 0].set_ylabel("Accuracy")
axs[1, 0].set_ylabel("Loss")

# Limites (ajuste conforme seu experimento)
# axs[0, 0].set_ylim(0.0, 0.6)
# axs[0, 1].set_ylim(0.0, 0.6)

# # Como sharex=True, definir xlim em um topo já propaga
# axs[0, 0].set_xlim(0, 50)

for ax in axs.ravel():
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=1, fontsize=9)

plt.tight_layout()

# Salvar figura
os.makedirs("figures", exist_ok=True)
out_path = os.path.join("figures", f"plot_acc_loss_vs_rounds_{name_figure}.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"[ok] figura salva em: {out_path}")
