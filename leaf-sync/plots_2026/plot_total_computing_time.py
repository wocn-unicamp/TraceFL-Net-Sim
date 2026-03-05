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

SIM_TYPE = "serial"  # "paralelo" | "serial" | "serial_lowcap"
FOLDER = f"../results/sys/fine_{SIM_TYPE}/"
OUT = f"figures/totalComputingTime/{SIM_TYPE}"
C = 64

# Escala: se o CSV tem ~100 rounds e você quer estimar 1000 rounds
SCALE_TO_1000 = 10

# Rounds (na escala 1000) para atingir accuracy X (vamos dividir por 10 para mapear nos ~100 rounds do CSV)
acc_rounds_1000 = {
    1: {"round_to_65": 320.0, "round_to_70": 440.0, "round_to_74": 680.0},
    2: {"round_to_65": 180.0, "round_to_70": 240.0, "round_to_74": 400.0},
    3: {"round_to_65": 120.0, "round_to_70": 180.0, "round_to_74": 360.0},
    4: {"round_to_65": 100.0, "round_to_70": 140.0, "round_to_74": 300.0},
    5: {"round_to_65":  80.0, "round_to_70": 120.0, "round_to_74": 280.0},
}

os.makedirs(OUT, exist_ok=True)

# ============================================================
# Helper: retorna o end_time (cumsum) no round mais próximo <= target
# ============================================================
def end_time_at_or_before_round(end_times: pd.Series, target_round: int) -> float:
    """
    end_times: Series index = round_number (int), value = tempo acumulado até o fim do round.
    """
    if end_times is None or len(end_times) == 0:
        return np.nan

    rounds = end_times.index.to_numpy(dtype=int)
    candidates = rounds[rounds <= target_round]
    chosen = int(candidates[-1]) if len(candidates) > 0 else int(rounds[0])
    return float(end_times.loc[chosen])


# ============================================================
# Plot 1: 4 barras por época (65/70/74 + 1000 rounds) em minutos
# ============================================================
def plot_training_time_bars(epochs, t65_sec, t70_sec, t74_sec, t1000_sec, scenario=""):
    plt.figure(figsize=(11, 6))

    x = np.arange(len(list(epochs)))
    width = 0.20

    # segundos -> minutos (estimado para 1000 rounds)
    t65_m   = (np.array(t65_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    t70_m   = (np.array(t70_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    t74_m   = (np.array(t74_sec, dtype=float)   * SCALE_TO_1000) / 60.0
    # t1000_m = (np.array(t1000_sec, dtype=float) * SCALE_TO_1000) / 60.0

    bars65   = plt.bar(x - 1.5 * width, t65_m,   width, label="Accuracy 65%")
    bars70   = plt.bar(x - 0.5 * width, t70_m,   width, label="Accuracy 70%")
    bars74   = plt.bar(x + 0.5 * width, t74_m,   width, label="Accuracy 74%")
    # bars1000 = plt.bar(x + 1.5 * width, t1000_m, width, label="1000 Rounds")

    plt.title("Total computing time to reach a target accuracy / 1000 Rounds in a" + f" ({scenario.capitalize()} Scenario)")
    plt.xlabel("Epochs")
    plt.ylabel("Time (min)")
    plt.ylim(1, 65)

    plt.xticks(x, list(epochs))
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    # Legenda del lado izquierdo superior
    plt.legend(loc="upper left")

    # ---- anotar valores nas barras (minutos), sempre acima ----
    ax = plt.gca()
    y_max = ax.get_ylim()[1]

    def annotate_bars(bar_container, values):
        for rect, v in zip(bar_container, values):
            if not np.isfinite(v):
                continue
            h = rect.get_height()
            y = min(h + 1.0, y_max - 1.0)
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                y,
                f"{v:.1f}",
                ha="center",
                va="bottom",
            )

    annotate_bars(bars65, t65_m)
    annotate_bars(bars70, t70_m)
    annotate_bars(bars74, t74_m)
    # annotate_bars(bars1000, t1000_m)

    plt.tight_layout()
    out_path = os.path.join(OUT, f"total_computing_time_targets_and_1000rounds_{scenario}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")


# ============================================================
# Plot 2: duração média do round (s) com IC 99.99%
# ============================================================
def plot_max_time_per_round(epochs, round_durations_per_epoch, confidence_intervals, scenario=""):
    plt.figure(figsize=(10, 6))

    means = [np.nan if len(v) == 0 else float(np.mean(v)) for v in round_durations_per_epoch]

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

    plt.title("Round duration with CI of 99.99%" + f" ({scenario.capitalize()} Scenario)")
    plt.xlabel("Epochs")
    plt.ylabel("Time [sec]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUT, f"max_time_per_round_with_confidence_interval_{scenario}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado em: {out_path}")


# ============================================================
# Main: só lê CSV e gera as métricas para as figuras
# ============================================================


scenarios = ["hom", "het"]  # "hom" = cenário homogêneo, "het" = cenário heterogêneo (com os tempos reais do CSV) --- "hom" é o foco principal, "het" é só para comparação



for scenario in scenarios:
    epochs = range(1, 6)
    t_total_100rounds_sec = []
    t65_sec, t70_sec, t74_sec = [], [], []
    round_durations_per_epoch = []
    confidence_intervals = []
    for EPOCH in epochs:
        path = os.path.join(FOLDER, f"sys_metrics_fedavg_c_{C}_e_{EPOCH}_time.csv")
        if not os.path.exists(path):
            print(f"[WARN] Arquivo não encontrado: {path}")
            t_total_100rounds_sec.append(np.nan)
            t65_sec.append(np.nan); t70_sec.append(np.nan); t74_sec.append(np.nan)
            round_durations_per_epoch.append([])
            confidence_intervals.append((0.0, 0.0))
            continue

        df_epoch = pd.read_csv(path)

        # ---- normalizar tipos ----
        df_epoch["computingTime_" + scenario] = pd.to_numeric(df_epoch.get("computingTime_" + scenario), errors="coerce")
        df_epoch["round_number"]  = pd.to_numeric(df_epoch.get("round_number"), errors="coerce")

        # ---- limpar linhas inválidas ----
        df_epoch = df_epoch.dropna(subset=["computingTime_" + scenario, "round_number"]).copy()
        df_epoch["round_number"] = df_epoch["round_number"].astype(int)
        df_epoch = df_epoch[df_epoch["computingTime_" + scenario] > 0].copy()

        if df_epoch.empty:
            print(f"[WARN] Época {EPOCH}: DataFrame vazio após limpeza.")
            t_total_100rounds_sec.append(np.nan)
            t65_sec.append(np.nan); t70_sec.append(np.nan); t74_sec.append(np.nan)
            round_durations_per_epoch.append([])
            confidence_intervals.append((0.0, 0.0))
            continue

        # (1) duração do round = max(computingTime) por round
        durations = df_epoch.groupby("round_number")["computingTime_" + scenario].max().sort_index()
        dur_list = durations.to_numpy(dtype=float).tolist()
        round_durations_per_epoch.append(dur_list)

        # (2) tempo acumulado até o fim de cada round (end_time)
        end_times = durations.cumsum()

        total_100_sec = float(end_times.iloc[-1])
        t_total_100rounds_sec.append(total_100_sec)

        # (3) tempo para targets de accuracy (mapeando para ~100 rounds)
        r65_100 = int(round(acc_rounds_1000[EPOCH]["round_to_65"] / SCALE_TO_1000))
        r70_100 = int(round(acc_rounds_1000[EPOCH]["round_to_70"] / SCALE_TO_1000))
        r74_100 = int(round(acc_rounds_1000[EPOCH]["round_to_74"] / SCALE_TO_1000))

        t65_sec.append(end_time_at_or_before_round(end_times, r65_100))
        t70_sec.append(end_time_at_or_before_round(end_times, r70_100))
        t74_sec.append(end_time_at_or_before_round(end_times, r74_100))

        # (4) IC 99.99% para duração do round
        if len(dur_list) >= 2:
            mean = float(np.mean(dur_list))
            ci = stats.t.interval(0.9999, len(dur_list) - 1, loc=mean, scale=stats.sem(dur_list))
            # guardar como (mean - lower, upper - mean) para usar como erro +/- no plot
            confidence_intervals.append((mean - ci[0], ci[1] - mean))
        else:
            confidence_intervals.append((0.0, 0.0))


    plot_training_time_bars(epochs, t65_sec, t70_sec, t74_sec, t_total_100rounds_sec, scenario=scenario)
    plot_max_time_per_round(epochs, round_durations_per_epoch, confidence_intervals, scenario=scenario)