import pandas as pd
import matplotlib.pyplot as plt

# Carregar o novo arquivo enviado (com sufixo x)
folder_path = "../results/stat/"
file_name = "stat_metrics_minibatch_c_3_mb_0.1.csv"

file_path = folder_path + file_name

df = pd.read_csv(file_path)

# Separar métricas de treino e teste
df_test = df[df["set"] == "test"]
df_train = df[df["set"] == "train"]

# Calcular métricas médias por round
acc_test = df_test.groupby("round_number")["accuracy"].mean()
acc_train = df_train.groupby("round_number")["accuracy"].mean()
loss_test = df_test.groupby("round_number")["loss"].mean()
loss_train = df_train.groupby("round_number")["loss"].mean()

# Plotar accuracy e loss em subplots
fig, axs = plt.subplots(2, 1, figsize=(10,10), sharex=True)

# Accuracy
axs[0].plot(acc_test.index, acc_test.values, marker='o', label="Test Accuracy")
axs[0].plot(acc_train.index, acc_train.values, marker='s', label="Train Accuracy")
axs[0].set_ylabel("Accuracy")
axs[0].set_title("Accuracy vs Rounds (Shakespeare FedAvg)")
axs[0].legend()
axs[0].grid(True)

# Loss
axs[1].plot(loss_test.index, loss_test.values, marker='o', label="Test Loss")
axs[1].plot(loss_train.index, loss_train.values, marker='s', label="Train Loss")
axs[1].set_xlabel("Rounds")
axs[1].set_ylabel("Loss")
axs[1].set_title("Loss vs Rounds (Shakespeare FedAvg)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()

# save the figure in the folder figures/
plt.savefig("figures/plot_acc_loss_vs_rounds_shakespeare_fedavg_c_8_e_1.png")

plt.show()