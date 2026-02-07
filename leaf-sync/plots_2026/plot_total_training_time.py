import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =========================
# Fonts (global)
# =========================
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

SIM_TYPE = "serial"  # "paralelo" ou "serial"
FOLDER = f"../results/sys/fine_{SIM_TYPE}/"
OUT = f"figures/trainingTime/{SIM_TYPE}"
C = 64

# Escala: si tus CSV tienen ~100 rounds y quieres estimar 1000 rounds
SCALE_TO_1000 = 10

# Rounds (en escala 1000) para alcanzar accuracy X (se dividirán entre 10)
acc_rounds_1000 = {
    1: {"round_to_65": 320.0, "round_to_70": 440.0, "round_to_74": 680.0},
    2: {"round_to_65": 180.0, "round_to_70": 240.0, "round_to_74": 400.0},
    3: {"round_to_65": 120.0, "round_to_70": 180.0, "round_to_74": 360.0},
    4: {"round_to_65": 100.0, "round_to_70": 140.0, "round_to_74": 300.0},
    5: {"round_to_65":  80.0, "round_to_70": 120.0, "round_to_74": 280.0},
}

os.makedirs(OUT, exist_ok=True)

# ============================================================
# Helper: end time (max time) en el round más cercano <= target
# ============================================================
def end_time_at_or_before_round(df_epoch, target_round: int) -> float:
    rounds = np.sort(df_epoch["round_number"].unique())
    if len(rounds) == 0:
        return np.nan

    candidates = rounds[rounds <= target_round]
    if len(candidates) == 0:
        chosen = int(rounds[0])
    else:
        chosen = int(candidates[-1])

    return float(df_epoch.loc[df_epoch["round_number"] == chosen, "time"].max())


# ============================================================
# Plot 1: 4 barras por época (65/70/74 + 1000 rounds) en horas
# ============================================================
def plot_training_time_bars(epochs, t65_sec, t70_sec, t74_sec, t1000_sec):
    plt.figure(figsize=(11, 6))

    x = np.arange(len(list(epochs)))
    width = 0.20

    # segundos -> minutos (estimación a 1000 rounds)
    t65_m   = (np.array(t65_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    t70_m   = (np.array(t70_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    t74_m   = (np.array(t74_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    t1000_m = (np.array(t1000_sec, dtype=float) * SCALE_TO_1000) / 60.0

    bars65   = plt.bar(x - 1.5 * width, t65_m,   width, label="Accuracy 65%")
    bars70   = plt.bar(x - 0.5 * width, t70_m,   width, label="Accuracy 70%")
    bars74   = plt.bar(x + 0.5 * width, t74_m,   width, label="Accuracy 74%")
    bars1000 = plt.bar(x + 1.5 * width, t1000_m, width, label="1000 Rounds")

    plt.title("Total Training Time by Target Accuracies and 1000 Rounds")
    plt.xlabel("Epochs")
    plt.ylabel("Time [minutes]")
    plt.ylim(1, 130)

    plt.xticks(x, list(epochs))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    # ---------- Etiquetas en cada barra (minutos) ----------
    ax = plt.gca()
    y_max = ax.get_ylim()[1]

    def annotate_bars(bar_container, values):
        for rect, v in zip(bar_container, values):
            if not np.isfinite(v):
                continue

            h = rect.get_height()

            # siempre arriba
            y = min(h + 1.0, y_max - 1.0)
            va = "bottom"

            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                f"{v:.1f}",
                ha="center",
                va=va,
            )


    annotate_bars(bars65, t65_m)
    annotate_bars(bars70, t70_m)
    annotate_bars(bars74, t74_m)
    annotate_bars(bars1000, t1000_m)
    # ------------------------------------------------------

    plt.tight_layout()
    out_path = os.path.join(OUT, "training_time_targets_and_1000rounds.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {out_path}")

# ============================================================
# Plot 2: Duración promedio del round (sec) con CI 99.99%
# ============================================================
def plot_max_time_per_round(epochs, round_durations_per_epoch, confidence_intervals):
    plt.figure(figsize=(10, 6))

    means = []
    for times in round_durations_per_epoch:
        means.append(np.nan if len(times) == 0 else float(np.mean(times)))

    lower_bound, upper_bound = [], []
    for mean, ci in zip(means, confidence_intervals):
        if np.isnan(mean):
            lower_bound.append(np.nan)
            upper_bound.append(np.nan)
        else:
            lower_bound.append(mean - ci[0])
            upper_bound.append(mean + ci[1])

    plt.bar(list(epochs), means, label="Mean Round Duration")

    yerr_low = np.array(means) - np.array(lower_bound)
    yerr_up  = np.array(upper_bound) - np.array(means)

    plt.errorbar(
        list(epochs), means,
        yerr=[yerr_low, yerr_up],
        fmt="o", color="black",
        label="Confidence Interval",
        capsize=5
    )

    plt.title("Round duration with CI of 99.99%")
    plt.xlabel("Epochs")
    plt.ylabel("Time [sec]")

    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT, "max_time_per_round_with_confidence_interval.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {out_path}")


# ============================================================
# Main
# ============================================================
epochs = range(1, 6)

t_total_100rounds_sec = []
t65_sec, t70_sec, t74_sec = [], [], []

round_durations_per_epoch = []
confidence_intervals = []

for EPOCH in epochs:
    path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}.csv")
    df_epoch = pd.read_csv(path)

    df_epoch["computingTime"] = pd.to_numeric(df_epoch["computingTime"], errors="coerce")
    df_epoch["round_number"]  = pd.to_numeric(df_epoch["round_number"], errors="coerce")

    df_epoch = df_epoch.dropna(subset=["computingTime", "round_number"]).copy()
    df_epoch["round_number"] = df_epoch["round_number"].astype(int)
    df_epoch = df_epoch[df_epoch["computingTime"] > 0].copy()

    if df_epoch.empty:
        t_total_100rounds_sec.append(np.nan)
        t65_sec.append(np.nan); t70_sec.append(np.nan); t74_sec.append(np.nan)
        round_durations_per_epoch.append([])
        confidence_intervals.append((0.0, 0.0))

        output_path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}_with_time.csv")
        df_epoch.to_csv(output_path, index=False)
        print(f"[WARN] Época {EPOCH}: DataFrame vacío tras limpieza. Guardado: {output_path}")
        continue

    # (1) duration(round) = max(computingTime | round)
    durations = df_epoch.groupby("round_number")["computingTime"].max().sort_index()
    dur_list = durations.to_numpy(dtype=float).tolist()
    round_durations_per_epoch.append(dur_list)

    # (2) time acumulado: time = computingTime + end_time(round-1)
    end_times = durations.cumsum()

    offsets = end_times.to_numpy(dtype=float).copy()
    offsets = np.roll(offsets, 1)
    offsets[0] = 0.0

    offset_by_round = dict(zip(durations.index.to_numpy(), offsets))
    df_epoch["time"] = df_epoch["computingTime"] + df_epoch["round_number"].map(offset_by_round)

    total_100_sec = float(end_times.iloc[-1])
    t_total_100rounds_sec.append(total_100_sec)

    # (3) tiempos para accuracy targets
    r65_100 = int(round(acc_rounds_1000[EPOCH]["round_to_65"] / SCALE_TO_1000))
    r70_100 = int(round(acc_rounds_1000[EPOCH]["round_to_70"] / SCALE_TO_1000))
    r74_100 = int(round(acc_rounds_1000[EPOCH]["round_to_74"] / SCALE_TO_1000))

    t65_sec.append(end_time_at_or_before_round(df_epoch, r65_100))
    t70_sec.append(end_time_at_or_before_round(df_epoch, r70_100))
    t74_sec.append(end_time_at_or_before_round(df_epoch, r74_100))

    # (4) CI 99.99% para duración de round
    if len(dur_list) >= 2:
        mean = float(np.mean(dur_list))
        ci = stats.t.interval(0.9999, len(dur_list) - 1, loc=mean, scale=stats.sem(dur_list))
        confidence_intervals.append((mean - ci[0], ci[1] - mean))
    else:
        confidence_intervals.append((0.0, 0.0))

    # Guardar CSV con time
    output_path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}_with_time.csv")
    df_epoch.to_csv(output_path, index=False)
    print(f"Archivo guardado para época {EPOCH} en: {output_path}")

plot_training_time_bars(epochs, t65_sec, t70_sec, t74_sec, t_total_100rounds_sec)
plot_max_time_per_round(epochs, round_durations_per_epoch, confidence_intervals)
