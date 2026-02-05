import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FOLDER = "../results/sys/"
OUT = "figures/trainingTime"
C = 64

# Crear el directorio de salida para los gráficos
os.makedirs(OUT, exist_ok=True)

# Función para graficar el máximo valor de 'time' por época
def plot_max_time_per_epoch(epochs, max_times):
    plt.figure(figsize=(10, 6))

    # Escalar los valores de tiempo por 10
    scaled_max_times = (np.array(max_times) * 10) / 60 
    plt.bar(epochs, scaled_max_times, color='skyblue')

    # Títulos y etiquetas en inglés
    plt.title("Total Training Time in 1000 Rounds", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Time [hour]", fontsize=14)

    # Ajustar tamaño de la fuente de los ticks
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    # Agregar grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Guardar el gráfico en la carpeta OUT
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "max_time_per_epoch.png"), dpi=150)
    print(f"Gráfico guardado en: {os.path.join(OUT, 'max_time_per_epoch.png')}")


# Función para graficar el máximo valor de 'computingTime' por round en cada época con intervalo de confianza
def plot_max_time_per_round(epochs, max_times_per_epoch, confidence_intervals):
    plt.figure(figsize=(10, 6))

    # Extraer los valores de la media y los intervalos de confianza
    means = [np.mean(times) for times in max_times_per_epoch]
    lower_bound = [mean - ci[0] for mean, ci in zip(means, confidence_intervals)]
    upper_bound = [mean + ci[1] for mean, ci in zip(means, confidence_intervals)]

    # Graficar las barras con los valores medios
    plt.bar(epochs, means, color='skyblue', label='Mean Round Duration')

    # Graficar los intervalos de confianza
    plt.errorbar(epochs, means, yerr=[np.array(means) - np.array(lower_bound), np.array(upper_bound) - np.array(means)],
                 fmt='o', color='black', label='Confidence Interval', capsize=5)

    # Títulos y etiquetas en inglés
    plt.title("Round duration with CI of 99.99%", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Time [sec]", fontsize=14)

    # Ajustar tamaño de la fuente de los ticks
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)

    # Agregar grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Leyenda
    plt.legend()

    # Guardar el gráfico en la carpeta OUT
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "max_time_per_round_with_confidence_interval.png"), dpi=150)
    print(f"Gráfico guardado en: {os.path.join(OUT, 'max_time_per_round_with_confidence_interval.png')}")



# Listas para almacenar los valores máximos de 'time' para cada época
epochs = range(1, 5)
max_times = []
max_times_per_epoch = []
confidence_intervals = []

# Iterar sobre las 4 épocas
for EPOCH in epochs:
    max_times_per_round = []

    # Cargar el dataset para la época correspondiente
    path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}.csv")
    df_epoch = pd.read_csv(path)

    # Crear la nueva columna 'time' inicializando con el 'computingTime'
    df_epoch['time'] = df_epoch['computingTime']

    # Para cada round_number (2, 3, 4, ...) sumar el valor máximo del 'time' del round anterior
    for round_num in range(1, df_epoch['round_number'].max() + 1):
        max_computing_time_in_round = df_epoch[df_epoch['round_number'] == round_num]['time'].max()
        max_times_per_round.append(max_computing_time_in_round)

    # Calcular el intervalo de confianza (95%) para el máximo de cada round
    mean = np.mean(max_times_per_round)
    ci = stats.t.interval(0.9999, len(max_times_per_round)-1, loc=mean, scale=stats.sem(max_times_per_round))
    confidence_intervals.append((mean - ci[0], ci[1] - mean))  # (lower, upper)

    # Almacenar los valores máximos por ronda para cada época
    max_times_per_epoch.append(max_times_per_round)

    # Obtener el valor máximo de 'time' para la época actual
    max_time = df_epoch['time'].max()
    max_times.append(max_time)

    # Guardar el DataFrame con la nueva columna 'time' para cada época
    output_path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}_with_time.csv")
    df_epoch.to_csv(output_path, index=False)

    # Confirmación para cada época
    print(f"Archivo guardado para época {EPOCH} en: {output_path}")

# Llamar a la función para crear el gráfico con los valores máximos de 'time'
plot_max_time_per_epoch(epochs, max_times)

# Llamar a la función para crear el gráfico con los valores máximos de 'time' por round y con intervalo de confianza
plot_max_time_per_round(epochs, max_times_per_epoch, confidence_intervals)
